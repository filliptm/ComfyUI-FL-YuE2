from contextlib import contextmanager

import torch
from torch import nn
from comfy.ldm.audio.autoencoder import OobleckDecoder, DecoderBlock, ResidualUnit, SnakeBeta
from comfy.model_management import throw_exception_if_processing_interrupted


class GrownDecoder(OobleckDecoder):
    """YuE2-VAE-RS x3 (Ruach Studio): the YuE2-Vae decoder with three residual units at the latent rate after the first
    convolution and three more at the end of each block, dilations 1, 3, 9 at the stream's width. Same latents as YuE2-Vae."""

    def __init__(self, **config):
        super().__init__(**config)
        snake, antialias = config.get("use_snake", False), config.get("antialias_activation", False)
        width = self.layers[0].out_channels
        self.pre = nn.Sequential(*(ResidualUnit(width, width, dilation, snake, antialias) for dilation in (1, 3, 9)))
        for block in self.layers:
            if isinstance(block, DecoderBlock):
                width = block.layers[1].out_channels
                block.layers.extend(ResidualUnit(width, width, dilation, snake, antialias) for dilation in (1, 3, 9))

    def forward(self, x):
        return self.layers[1:](self.pre(self.layers[0](x)))


def grown_weights(weights):
    """YuE2-VAE-RS release tensor names -> GrownDecoder's (ComfyUI Oobleck names plus decoder.pre)."""
    unit = ("snake1", "conv1", "snake2", "conv2")
    renamed = {}
    for key, value in weights.items():
        part = key.split(".")
        if part[0] in ("conv1", "snake1", "conv2"):
            name = ["layers", {"conv1": 0, "snake1": 7, "conv2": 8}[part[0]], *part[1:]]
        elif part[0] == "pre" and part[2] in unit:
            name = ["pre", part[1], "layers", unit.index(part[2]), *part[3:]]
        elif part[0] == "blocks" and part[2] in ("snake1", "conv_t1"):
            name = ["layers", int(part[1]) + 1, "layers", ("snake1", "conv_t1").index(part[2]), *part[3:]]
        elif part[0] == "blocks" and part[2] == "units" and part[4] in unit:
            name = ["layers", int(part[1]) + 1, "layers", int(part[3]) + 2, "layers", unit.index(part[4]), *part[5:]]
        else:
            raise ValueError(f"Unexpected YuE2-VAE-RS tensor: {key}")
        renamed["decoder." + ".".join(map(str, name))] = value
    return renamed


@contextmanager
def exact_float32(device):
    """cuDNN may run float32 convolutions as TF32 on Ampere and newer; the x3 decoders are validated without it."""
    if device.type != "cuda":
        yield
        return
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def _stages(module):
    if isinstance(module, nn.Sequential):
        return list(module)
    layers = list(module.layers)
    return layers[:1] + [module.pre] + layers[1:] if isinstance(module, GrownDecoder) else layers


def _dependency_interval(module, low, high):
    """Inclusive input support of an output interval; no waveform blending."""
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        for child in reversed(_stages(module)):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, ResidualUnit):
        a, b = _dependency_interval(module.layers, low, high)
        return min(a, low), max(b, high)
    if isinstance(module, nn.ConvTranspose1d):
        s, p, d, k = (module.stride[0], module.padding[0],
                      module.dilation[0], module.kernel_size[0])
        return -(-(low + p - d * (k - 1)) // s), (high + p) // s
    if isinstance(module, nn.Conv1d):
        s, p, d, k = (module.stride[0], module.padding[0],
                      module.dilation[0], module.kernel_size[0])
        return low * s - p, high * s - p + d * (k - 1)
    if isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    raise TypeError(f"No audited support rule for {type(module).__name__}")


def _output_length(module, length):
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        for child in _stages(module):
            length = _output_length(child, length)
        return length
    if isinstance(module, nn.ConvTranspose1d):
        return ((length - 1) * module.stride[0] - 2 * module.padding[0]
                + module.dilation[0] * (module.kernel_size[0] - 1)
                + module.output_padding[0] + 1)
    if isinstance(module, nn.Conv1d):
        return ((length + 2 * module.padding[0]
                 - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
                // module.stride[0] + 1)
    if isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    raise TypeError(f"No audited length rule for {type(module).__name__}")


class YuE2VAE(nn.Module):
    def __init__(self, config, grown=False):
        super().__init__()
        decoder = dict(config["decoder_config"])
        if decoder.pop("snake_type", "vanilla") != "vanilla" or decoder.pop("use_filter", False):
            raise ValueError("Unsupported YuE2 decoder format")
        self.decoder = (GrownDecoder if grown else OobleckDecoder)(**decoder)
        self.exact = grown
        self.ratio = config["downsampling_ratio"]
        self.sample_rate = config["sample_rate"]

    def decode(self, latent):
        if not self.exact:
            return self.decoder(latent)
        with exact_float32(latent.device):
            return self.decoder(latent)

    def decode_tiled(self, latent, core_frames=1024, on_progress=None):
        frames = latent.shape[-1]
        total = _output_length(self.decoder, frames)
        low, high = _dependency_interval(self.decoder, 0, core_frames * self.ratio - 1)
        halo = max(16, -low, high - core_frames + 1)
        output = torch.empty((latent.shape[0], 2, total), dtype=torch.float32, device="cpu")
        tiles = (frames + core_frames - 1) // core_frames
        for index, start in enumerate(range(0, frames, core_frames)):
            throw_exception_if_processing_interrupted()
            end = min(frames, start + core_frames)
            left, right = max(0, start - halo), min(frames, end + halo)
            device = next(self.decoder.parameters()).device
            tile = self.decode(latent[..., left:right].to(device=device, dtype=torch.float32))
            out_start, out_end = start * self.ratio, min(end * self.ratio, total)
            crop_start = (start - left) * self.ratio
            output[..., out_start:out_end].copy_(tile[..., crop_start:crop_start + out_end - out_start])
            del tile
            if on_progress is not None:
                on_progress(index + 1, tiles)
        return output
