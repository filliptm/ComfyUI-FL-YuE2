# Third-party code notices

The Oobleck VAE and SnakeBeta components reused from ComfyUI are derived
from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45`.
The module hierarchy, weight normalization and activation equations preserve
the checkpoint's original inference implementation.

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full text: `licenses/stable-audio-tools-MIT.txt`.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full text: `licenses/SnakeBeta-NVIDIA-MIT.txt`.

These notices cover the identified source code and retain its original licenses.
The YuE2 model checkpoint weights are separately licensed under CC BY-NC 4.0;
see MODEL_LICENSE for the scope and full terms. This does not relicense third-party code.

The model, protocol, sampling, acoustic synthesis, tokenizer and VAE tiling
are adapted from YuE2 commit 92a73cc7652fcc1f937855e4b765e0a0edd7ff2e.
Copyright (c) 2026 the YuE2 authors. Apache License 2.0; see LICENSE.

abc_score.py adapts the native ABC parser from the same commit's
skills/yue2-music/scripts/abc_tools.py, under the same Apache 2.0 license.

The optional YuE2-VAE-RS x3 decoder layout (three residual units at the latent
rate and three more per decoder block) follows Ruach Studio's YuE2-VAE-RS
release, https://huggingface.co/goldhub/YuE2-RS-Finetunes (integration code
Apache 2.0, Copyright 2026 Viktor Zhuromskyy, Ruach Studio). Its weights are
not bundled; they download on request under CC BY-NC 4.0 (see MODEL_LICENSE).
