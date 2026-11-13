import hashlib

import pytest
import torch
from safetensors.torch import save_file

from fl_yue2.yue2 import downloads
from fl_yue2.yue2.nodes import FL_YuE2_DecoderLoader
from fl_yue2.yue2.vae import YuE2VAE, _dependency_interval, exact_float32, grown_weights

CONFIG = {"downsampling_ratio": 1920, "sample_rate": 48000, "decoder_config": {
    "c_mults": [1, 2, 4, 8, 16, 32], "channels": 2, "final_tanh": False, "latent_dim": 64, "out_channels": 2,
    "snake_type": "vanilla", "strides": [2, 2, 4, 4, 5, 6], "use_filter": False, "use_snake": True}}
CONV = ("bias", "parametrizations.weight.original0", "parametrizations.weight.original1")
UNIT = ("snake1.alpha", "snake1.beta", *(f"conv1.{leaf}" for leaf in CONV), "snake2.alpha", "snake2.beta", *(f"conv2.{leaf}" for leaf in CONV))


def release_names():
    names = [f"conv1.{leaf}" for leaf in CONV] + [f"pre.{i}.{leaf}" for i in range(3) for leaf in UNIT]
    for block in range(6):
        names += [f"blocks.{block}.snake1.alpha", f"blocks.{block}.snake1.beta"] + [f"blocks.{block}.conv_t1.{leaf}" for leaf in CONV]
        names += [f"blocks.{block}.units.{unit}.{leaf}" for unit in range(6) for leaf in UNIT]
    return names + ["snake1.alpha", "snake1.beta", "conv2.parametrizations.weight.original0", "conv2.parametrizations.weight.original1"]


def seeded(vae, seed=3):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in vae.state_dict(keep_vars=True).items():
            value.copy_(torch.randn(value.shape, generator=generator) * (0.05 if "alpha" in name or "beta" in name else 0.3))
    return vae.eval()


def test_release_names_fill_the_grown_decoder_exactly():
    names = release_names()
    assert len(names) == 427
    with torch.device("meta"):
        vae = YuE2VAE(CONFIG, grown=True)
    assert set(grown_weights(dict.fromkeys(names))) == set(vae.state_dict())
    with pytest.raises(ValueError, match="Unexpected"):
        grown_weights({"encoder.layers.0.bias": None})


def test_grown_halo_covers_new_units_and_tiles_match_whole_decode():
    standard, grown = seeded(YuE2VAE(CONFIG)), seeded(YuE2VAE(CONFIG, grown=True))
    halos = []
    for vae in (standard, grown):
        low, high = _dependency_interval(vae.decoder, 0, 1024 * 1920 - 1)
        halos.append(max(-low, high - 1023))
    assert halos == [12, 60]
    latent = torch.randn(2, 64, 150, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        whole = grown.decode(latent)
        assert whole.shape == (2, 2, 150 * 1920)
        for core in (64, 100, 150):
            torch.testing.assert_close(grown.decode_tiled(latent, core), whole, atol=2e-5, rtol=1e-5)


def test_exact_float32_restores_tf32_flags():
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        with exact_float32(torch.device("cpu")):
            assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
        with pytest.raises(RuntimeError):
            with exact_float32(torch.device("cuda")):
                assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                raise RuntimeError
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def release_file(path, step="1", dtype=torch.float32, drop=0, extra=None, scale=1.0):
    names = release_names()[drop:]
    tensors = {name: torch.full((2,), scale, dtype=dtype) for name in names}
    tensors.update(extra or {})
    save_file(tensors, str(path), metadata={"variant": "x3", "step": step})
    return path.read_bytes()


def test_pinned_decoder_download_and_refusal_reasons(monkeypatch, tmp_path):
    release = release_file(tmp_path / "release.safetensors")
    monkeypatch.setattr(downloads, "DECODERS", {"YuE2-VAE-RS-x3_v1": (hashlib.sha256(release).hexdigest(), "1"),
                                                "YuE2-VAE-RS-x3_v2": ("0" * 64, "2")})
    monkeypatch.setattr(downloads.folder_paths, "get_folder_paths", lambda _: [str(tmp_path)])
    calls = []
    def transfer(url, target):
        calls.append(url.rsplit("/", 1)[1])
        target.write_bytes(release if target.suffix == ".safetensors" else b"license")
    monkeypatch.setattr(downloads, "transfer_url", transfer)
    with pytest.raises(FileNotFoundError, match="download_missing"):
        downloads.decoder("YuE2-VAE-RS-x3_v1", False)
    target = downloads.decoder("YuE2-VAE-RS-x3_v1")
    assert target == tmp_path / "YuE2-VAE-RS" / "YuE2-VAE-RS-x3_v1.safetensors"
    assert calls == ["LICENSE", "YuE2-VAE-RS-x3_v1.safetensors"]
    assert downloads.decoder("YuE2-VAE-RS-x3_v1", False) == target and len(calls) == 2
    first = FL_YuE2_DecoderLoader.IS_CHANGED("YuE2-VAE-RS-x3_v1")
    assert FL_YuE2_DecoderLoader.IS_CHANGED("YuE2-VAE-RS-x3_v1") == first
    copies = {"float32 only": {"dtype": torch.float16}, "incomplete \\(426 of 427": {"drop": 1},
              "-full file": {"extra": {"encoder.layers.0.bias": torch.zeros(2)}}, "x3_v2 release": {"step": "2"},
              "modified copy": {"scale": 2.0}}
    for reason, change in copies.items():
        replacement = tmp_path / "replacement.safetensors"
        release_file(replacement, **change)
        replacement.replace(target)
        assert FL_YuE2_DecoderLoader.IS_CHANGED("YuE2-VAE-RS-x3_v1") != first
        with pytest.raises(ValueError, match=reason):
            downloads.decoder("YuE2-VAE-RS-x3_v1", False)
    save_file({"encoder.layers.0.bias": torch.zeros(2), "decoder.layers.0.bias": torch.zeros(2)}, str(target))
    with pytest.raises(ValueError, match="not a YuE2-VAE-RS decoder"):
        downloads.decoder("YuE2-VAE-RS-x3_v1", False)
    target.write_bytes(b"not safetensors")
    with pytest.raises(ValueError, match="not a readable"):
        downloads.decoder("YuE2-VAE-RS-x3_v1", False)
    target.write_bytes(release)
    assert downloads.decoder("YuE2-VAE-RS-x3_v1", False) == target
    assert FL_YuE2_DecoderLoader.IS_CHANGED("YuE2-Vae") == "YuE2-Vae"
    with pytest.raises(ValueError, match="Unknown"):
        downloads.decoder("../YuE2-VAE-RS-x3_v1")
