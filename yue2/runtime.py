from dataclasses import dataclass
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file
import comfy.model_management as mm
import comfy.ops
from comfy.model_patcher import ModelPatcher
from comfy.utils import ProgressBar

from .downloads import resolve, decoder
from .model import YuE2Model
from .vae import YuE2VAE, grown_weights
from .tokenizer import YuE2TextTokenizer
from .protocol import SongRequest, GenerationConfig, token_prefixes, negative_prefix, CODEC_OFFSET, resolve_sampling
from .sampling import generate_tokens
from .nar import synthesize


@dataclass(frozen=True)
class Plan:
    request: SongRequest
    abc: str | None
    abc_ids: list[int]
    prefix: list[int]
    truncated: bool = False


@dataclass
class MusicModel:
    patcher: ModelPatcher
    tokenizer: YuE2TextTokenizer
    generation: GenerationConfig

    def prepare(self, tokens, branches=1):
        c = self.patcher.model.config
        kv_bytes = 2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * tokens * 2 * branches
        mm.load_models_gpu([self.patcher], memory_required=kv_bytes + 2 * 1024**3, force_full_load=True)
        model = self.patcher.model
        for layer in model.modules():
            if isinstance(layer, comfy.ops.manual_cast.Linear) and layer.weight.device == self.patcher.load_device:
                layer.comfy_cast_weights = False
        return model


def load_models(download_missing=True, model_directory=None):
    device = mm.get_torch_device()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("FL YuE2 currently requires an NVIDIA GPU with BF16 support.")
    paths = [Path(model_directory) if model_directory else resolve("YuE2-3B", download_missing), resolve("YuE2-Vae", download_missing)]
    config = json.loads((paths[0] / "config.json").read_text())
    with torch.device("meta"):
        model = YuE2Model(config)
    model.load_state_dict(load_file(str(paths[0] / "model.safetensors")), strict=True, assign=True)
    model.eval()
    patcher = ModelPatcher(model, device, mm.unet_offload_device())
    generation = GenerationConfig.from_dict(json.loads((paths[0] / "yue2_generation_config.json").read_text()))
    music = MusicModel(patcher, YuE2TextTokenizer(paths[0] / "qwen.tiktoken"), generation)

    return music, load_vae(paths[1])


def load_vae(root, decoder_file=None):
    """YuE2-Vae's decoder, or with decoder_file a YuE2-VAE-RS x3 decoder for the same latents (YuE2-Vae supplies its config)."""
    config = json.loads((root / "config.json").read_text())
    with torch.device("meta"):
        vae = YuE2VAE(config, grown=decoder_file is not None)
    if decoder_file is None:
        weights = load_file(str(root / "model.safetensors"))
        weights = {key.replace(".weight_g", ".parametrizations.weight.original0").replace(".weight_v", ".parametrizations.weight.original1"): value
                   for key, value in weights.items() if key.startswith("decoder.")}
    else:
        weights = grown_weights(load_file(str(decoder_file)))
    vae.load_state_dict(weights, strict=True, assign=True)
    vae.eval()
    return ModelPatcher(vae, mm.get_torch_device(), mm.unet_offload_device())


def load_decoder(name, download_missing=True):
    root = resolve("YuE2-Vae", download_missing)
    return load_vae(root, None if name == "YuE2-Vae" else decoder(name, download_missing))


def cancelled():
    mm.throw_exception_if_processing_interrupted()
    return False


def token_progress(total, on_progress=None):
    bar = ProgressBar(total)
    count = 0

    def update(phase, token):
        nonlocal count
        count += 1
        bar.update_absolute(count)
        if on_progress is not None:
            on_progress("tokens", count, total)
    return update


def make_plan(music, style, lyrics, seed, mode, abc, max_tokens):
    request = SongRequest(style=style, lyrics=lyrics, seed=seed, cot=mode, abc=abc.strip() or None)
    if mode == "off":
        return Plan(request, None, [], token_prefixes(request, music.tokenizer))
    if request.abc is not None:
        ids = music.tokenizer.encode(request.abc)
        return Plan(request, request.abc, ids, token_prefixes(request, music.tokenizer, ids))
    sampling = resolve_sampling({"max_tokens": max_tokens, "min_tokens": min(32, max_tokens)}, music.generation.abc)
    prefix = token_prefixes(request, music.tokenizer)
    model = music.prepare(len(prefix) + max_tokens)
    ids, timing, truncated = generate_tokens(model, prefix, sampling, seed, "abc", cancelled=cancelled, on_token=token_progress(max_tokens))
    if truncated:
        logging.warning("YuE2 score limit reached; continuing with the partial score.")
    return Plan(request, music.tokenizer.decode(ids), ids, token_prefixes(request, music.tokenizer, ids), truncated=truncated)


def render(music, plan, max_seconds, temperature, top_p, top_k, repetition_penalty, cfg_scale, steps, *, on_progress=None, check_cancelled=cancelled):
    if token_prefixes(plan.request, music.tokenizer, plan.abc_ids) != plan.prefix:
        raise ValueError("YuE2 plan changed. Submit edited ABC through the Plan node.")
    max_tokens = round(max_seconds * 25)
    sampling = resolve_sampling({"max_tokens": max_tokens, "min_tokens": min(200, max_tokens),
                                 "temperature": temperature, "top_p": top_p, "top_k": top_k,
                                 "repetition_penalty": repetition_penalty}, music.generation.semantic)
    negative = negative_prefix(plan.request, music.tokenizer, plan.abc_ids) if cfg_scale != 1 else None
    model = music.prepare(len(plan.prefix) + max_tokens, 1 if cfg_scale == 1 else 2)
    ids, timing, truncated = generate_tokens(model, plan.prefix, sampling, plan.request.seed, "semantic",
                                            negative=negative, cfg_scale=cfg_scale, legacy_off=plan.request.cot == "off",
                                            cancelled=check_cancelled, on_token=token_progress(max_tokens, on_progress))
    if not ids:
        raise ValueError("YuE2 produced no music tokens. Try a different seed or lyrics.")
    if truncated:
        logging.warning("YuE2 reached max_duration; increase it if the song ends early.")
    bar = ProgressBar(steps)
    def progress(done, total):
        bar.update_absolute(done, total)
        if on_progress is not None:
            on_progress("synthesis", done, total)
    if on_progress is not None:
        on_progress("synthesis", 0, steps)
    latent = synthesize(model, plan.prefix, [token - CODEC_OFFSET for token in ids], plan.request.seed,
                        steps=steps, cancelled=check_cancelled, on_progress=progress)
    # The runtime owns YuE2's native [B,C,T] layout.
    return latent.T.unsqueeze(0).contiguous(), truncated, timing


def decode(vae, latent, tile_frames, *, on_progress=None, check_cancelled=cancelled):
    if latent.ndim != 3 or latent.shape[1] != 64 or latent.shape[-1] == 0:
        raise ValueError("Expected YuE2 latents shaped [batch,64,frames]")
    mm.load_models_gpu([vae], memory_required=2 * 1024**3, force_full_load=True)
    bar = ProgressBar((latent.shape[-1] + tile_frames - 1) // tile_frames)
    def progress(done, total):
        check_cancelled()
        bar.update_absolute(done, total)
        if on_progress is not None:
            on_progress("decode", done, total)
    if on_progress is not None:
        on_progress("decode", 0, bar.total)
    audio = vae.model.decode_tiled(latent, tile_frames, progress)
    if not torch.isfinite(audio).all():
        raise ValueError("YuE2 produced non-finite audio")
    return {"waveform": audio.clamp_(-1, 1), "sample_rate": vae.model.sample_rate}
