import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

from filelock import FileLock
from safetensors import safe_open
import folder_paths
from comfy.utils import ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted


MODELS = {
    "YuE2-3B": "1a96eca688d6ae5d7f0feb88573fec89920fcd19",
    "YuE2-Vae": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
    "MERT-v2-FullSong": "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
}
# Ruach Studio's YuE2-VAE-RS x3 decoders (CC BY-NC 4.0): decoder-only float32 files, same latents as YuE2-Vae.
DECODER_REPOSITORY = "https://huggingface.co/goldhub/YuE2-RS-Finetunes/resolve/b371be93fe7320c7084813b4f5967a6820756a06/VAE-RS"
DECODERS = {
    "YuE2-VAE-RS-x3_v1": ("a88c8da0d0913115953196b01aaf770ea128ab224bde090a8f34ee46db3bb039", "161953"),
    "YuE2-VAE-RS-x3_v2": ("114da484c564a265a694f019d0715cde2f87da88815a054f72ce77c5250f1a0c", "155030"),
}
COMMON_FILES = ("config.json", "weights_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md",
                "licenses/SnakeBeta-NVIDIA-MIT.txt", "licenses/stable-audio-tools-MIT.txt")
folder_paths.add_model_folder_path("yue2", str(Path(folder_paths.models_dir) / "yue2"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            throw_exception_if_processing_interrupted()
            result.update(block)
    return result.hexdigest()


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("YuE2 model file escapes its model directory")
    return path


def transfer(name, revision, filename, target):
    transfer_url(f"https://huggingface.co/m-a-p/{name}/resolve/{revision}/{filename}", target)


def transfer_url(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink():
        raise ValueError("YuE2 partial download must not be a symlink")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(url,
                      headers={"Range": f"bytes={offset}-"} if offset else {})
    logging.info("YuE2: downloading %s", target.name)
    with urlopen(request, timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Unexpected download resume offset")
        completed = offset if resumed else 0
        total = completed + int(response.headers.get("Content-Length", 0))
        progress = ProgressBar(max(total, 1))
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                throw_exception_if_processing_interrupted()
                stream.write(block)
                completed += len(block)
                progress.update_absolute(completed, max(total, completed))
        if total and completed != total:
            raise IOError(f"Incomplete YuE2 download: {target.name}; queue again to resume")
    os.replace(partial, target)


def resolve(name, download=True):
    if name not in MODELS:
        raise ValueError("Unknown YuE2 model selection")
    revision = MODELS[name]
    files = list(COMMON_FILES)
    if name == "YuE2-3B":
        files += ["qwen.tiktoken", "generation_config.json", "yue2_generation_config.json"]
    elif name == "MERT-v2-FullSong":
        files = ["config.json", "preprocessor_config.json", "configuration_mert2.py", "modeling_mert2.py",
                 "weights_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md"]
    roots = folder_paths.get_folder_paths("yue2")
    candidates = [contained(root, name) for root in roots]
    target = next((path for path in candidates if all(contained(path, file).is_file() for file in files + ["model.safetensors"])),
                  next((path for path in candidates if path.is_dir()), candidates[0]))
    if not download and not target.is_dir():
        raise FileNotFoundError(f"Install {name} in {target} or enable download_missing")
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(contained(target, ".download.lock"))):
        for filename in files + ["model.safetensors"]:
            path = contained(target, filename)
            if not path.is_file():
                if not download:
                    raise FileNotFoundError(f"Missing {path}; enable download_missing to complete installation")
                transfer(name, revision, filename, path)
        manifest = json.loads((target / "weights_manifest.json").read_text())
        if name == "MERT-v2-FullSong":
            manifest = {"files": {manifest["filename"]: manifest}}
        if set(manifest["files"]) != {"model.safetensors"}:
            raise ValueError("Unexpected YuE2 weight manifest")
        weight = contained(target, "model.safetensors")
        stamp = contained(target, ".verified.json")
        state = {"revision": revision, "manifest": digest(target / "weights_manifest.json"),
                 "files": {filename: [contained(target, filename).stat().st_size,
                                      contained(target, filename).stat().st_mtime_ns]
                           for filename in files + ["model.safetensors"]}}
        previous = json.loads(stamp.read_text()) if stamp.is_file() else None
        if previous != state:
            expected = manifest["files"]["model.safetensors"]
            if weight.stat().st_size != expected["bytes"] or digest(weight) != expected["sha256"]:
                raise ValueError(f"Corrupt YuE2 weights: {weight}. Remove that file and queue again to download it.")
            temporary = contained(target, ".verified.tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            os.replace(temporary, stamp)
    return target


def decoder_path(name):
    if name not in DECODERS:
        raise ValueError("Unknown YuE2 decoder selection")
    candidates = [contained(Path(root) / "YuE2-VAE-RS", name + ".safetensors") for root in folder_paths.get_folder_paths("yue2")]
    return next((path for path in candidates if path.is_file()), candidates[0])


def decoder(name, download=True):
    path = decoder_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {path}; enable download_missing to download {name}")
            if not contained(path.parent, "LICENSE").is_file():
                transfer_url(f"{DECODER_REPOSITORY}/LICENSE", contained(path.parent, "LICENSE"))
            transfer_url(f"{DECODER_REPOSITORY}/{path.name}", path)
        if digest(path) != DECODERS[name][0]:
            raise ValueError(f"{path} is not the {name} release: {mismatch(path)}. Remove it and queue again to download it.")
    return path


def mismatch(path):
    """Why a file under a pinned decoder name is not that release; the digest has already disagreed."""
    try:
        with safe_open(str(path), framework="pt") as weights:
            keys, step = list(weights.keys()), (weights.metadata() or {}).get("step")
            dtypes = {weights.get_slice(key).get_dtype() for key in keys}
    except Exception as error:
        return f"not a readable safetensors file ({type(error).__name__})"
    if not any(key.startswith(("pre.", "blocks.")) for key in keys):
        return "it is not a YuE2-VAE-RS decoder"
    if any(key.startswith("encoder.") for key in keys):
        return "it is a -full file (encoder and decoder); Decode Audio needs the decoder-only release"
    if dtypes - {"F32"}:
        return f"it holds {', '.join(sorted(dtypes - {'F32'}))} tensors; the decoder is released and validated in float32 only"
    if len(keys) != 427:
        return f"it is incomplete ({len(keys)} of 427 tensors)"
    other = next((name for name, (_, release_step) in DECODERS.items() if release_step == step and path.stem != name), None)
    return f"it is the {other} release" if other else "it is a modified copy"
