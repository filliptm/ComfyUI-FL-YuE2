# YuE2-VAE-RS x3 decoders

Load Audio Decoder adds Ruach Studio's YuE2-VAE-RS x3 v1/v2 as optional decoders. Validated on CPU only (Linux, Python 3.12, Torch 2.15.0+cpu, ComfyUI c5751c8); GPU decoding, including the TF32-off path, has not been run for this change. The four new tests pass; the remaining failures are the same as on main in this environment (the captioner test needs google-genai, and two paired-audio training tests fail intermittently on CPU).

Both files downloaded from the pinned revision and matched their SHA-256. Against independent reference decodes by Ruach Studio's yue2_vae_rs.py, 84 in-process decodes (both versions; 1 to 3,000 latent frames, a batch of two included; tile_frames 64 to 2,048) differed by at most 2.1e-5 (repeat runs vary in the last digit with CPU threading), and 156 x3 decodes through ComfyUI's HTTP API by at most 2.0e-5. The derived context is 60 frames for x3 and still 12 (16 with the floor) for YuE2-Vae. YuE2-Vae through the refactored loader is bit-identical to main (24 decodes), and the training encoder's latents are bit-identical (three files, mono and stereo, 6 to 70 seconds). Files replaced under the same name were verified again on the next queue: v2 under the v1 name, float16, incomplete, -full, YuE2-Vae, truncated and modified files were refused with their reason, and the release decoded again once restored. Latents arriving in bfloat16, float16 or float64 decode exactly as their float32 values, and `--fp16-vae` / `--bf16-vae` servers gave the same differences from the reference as the default server, to every digit (12 runs).

# Caption acceptance and validation baselines

Generated captions no longer require manual review. Validation previews now include a separate step-0 baseline and phase progress. The baseline uses the run's starting model and acoustic companion, and is not an exported AR checkpoint.

Validation: 103 Python tests passed, including exact equality of resumed and uninterrupted two-step optimizer results, step-0 resume state before any update, caption acceptance, changed-audio rejection, baseline caching and changed preview settings. Browser DOM checks passed for baseline ordering/play controls, hidden baseline adapter selection, independent training/inference progress and cancellation cleanup. JavaScript syntax and diff whitespace checks passed.

A full-size GPU smoke run remains pending because the main ComfyUI instance was actively training during verification. The active job was not interrupted or restarted.

# Version 0.2.3

Version 0.2.3 repository cleanup: moved the Python implementation into `yue2/`, preserving node identifiers, web routes, model paths, and saved-run locations. All 99 existing tests pass, plus a new isolated-worker startup regression test. The restarted isolated ComfyUI server registers all 12 nodes; score validation and saved-run retrieval return HTTP 200. A fresh GPU inference through the relocated package produced 8 seconds of finite, non-silent stereo audio at 48 kHz. README images are actual browser captures of existing inference and training workflows, including a generation capped at 180 seconds and the saved 500-step AR run with checkpoint previews.

# Version 0.2.0

The release adds AR LoRA training, checkpoint previews during training, tooltips, and partial-score continuation. See [training validation](TRAINING_VALIDATION.md). Historical inference/editor checks follow.

# Validation — 2026-09-10

Tested on Windows, Python 3.12.6, Torch 2.11.0+cu130, Comfy Kitchen 0.2.31, RTX PRO 6000 Blackwell Max-Q (approximately 96 GB VRAM). The existing Python environment was preserved; no dependencies were installed or downgraded.

## Results

Multi-selection update: actual JavaScript modules with the DOM fixture and local ABC API passed Shift-drag selection in reverse direction, non-mutating selection, group outlines/previews, group pitch/time moves, shared boundary clamping, overlap rollback with selection retained, pointer cancellation, Shift-click toggling, group keyboard movement, deletion and undo at 50% canvas scale. Live-browser visual verification remains unavailable.

Latest editor simplification: 49 Python tests pass, including persisted connected edits, new-upstream replacement and invalid-edit rejection. Actual-module JavaScript checks cover drawing/moving/resizing, pitch audition, connected editing, preserved timing and intervals under transposition, slash-chord bass transposition, out-of-range rejection, pause/resume position, unsnapped scrubbing at 50% canvas scale, playback after scrubbing, and metronome beat/downbeat scheduling. Tests use a DOM fixture and mocked Web Audio with the real local ABC validation API; browser sound and visual appearance remain unverified.

Real-model connected editing passed on the isolated server: initial generated-score render produced a 15.9987-second stereo clip in 25.0 seconds; changing a note while retaining the input connection and source fingerprint produced `connected_edited_score_00001.flac` in 10.3 seconds. The second execution returned the exact edited ABC, and both outputs were finite, non-silent, 48 kHz stereo with a 16-second test cap. Earlier read-only/capture behavior described below is superseded.

Content sizing update: JavaScript syntax and actual-module DOM-fixture checks pass for content-driven node growth, resize clamping, decreasing the minimum height after content collapses, and fitting the timeline to a changed available width. Existing piano editing, persistence, connected-score and playback-scheduling checks also pass against the running local validation API. These checks simulate layout measurements; live-browser layout remains unverified.

Latest editor update: 47 Python tests pass, including connected-score precedence, UI score delivery and empty-input rejection. Actual editor JavaScript with a DOM fixture and local validation API passed connected/local draft isolation, capture/disconnection, undo/redo, reload, draw/move/resize at 50% canvas scale, keyboard transpose/nudge, zoom/fit and select-tool checks. Live browser appearance and browser audio playback remain unverified because no browser debugging connection is available.

The new generated-score branch completed real-model Compose → Piano Roll → Compose → Render → Decode → Save on the isolated local server in 28.4 seconds. It produced `output/yue2_validation/connected_editor_00001.flac`: finite, non-silent, 48 kHz stereo, 15.9987 seconds with a 16-second test cap. This validates score generation and pass-through into rendering, not exact musical adherence or the full uncapped song. The combined workflow now has five branches, 26 nodes and 34 checked links; unchanged branches were not re-rendered for this update.

Latest planning correction: 45 pack tests pass. Blank and whitespace-only lyrics now preserve explicit `full`/`melody` planning; only `off` skips it. Regression tests verify the ABC generation path, supplied-score reuse, and a clear token-limit error without automatic fallback. The earlier blank-lyrics direct-generation behavior recorded below is superseded. Real-model instrumental score completion has not been revalidated for this correction.

- Both pinned model downloads completed and passed SHA-256/size verification; subsequent loads used local files.
- Upstream small-model baseline: 59 passed, one skipped.
- 44 pack tests passed. In addition to model, download and native ABC checks, piano-roll tests verify exact pitch/timing/chord round trips, ties across bars and chord changes, short notes, adjacent attacks, pitch-range checks, key changes preserving sounding notes, and overlap/out-of-bounds rejection. Ruff and JavaScript syntax checks passed.
- Piano-roll interaction tests exercised the actual JavaScript modules with a DOM fixture and the real local API: drawing, dragging pitch/time, resizing at 50% canvas scale, deleting notes, section duplication/reordering/removal, adding bars, initialization, hidden source storage, undo/redo and configuration reload. These are event and persistence tests, not live visual browser tests.
- Adapted AR prefill, cached token decode, and acoustic velocities matched the upstream small-model reference within numerical tolerances on CPU and CUDA. Reused ComfyUI Oobleck decoding matched the upstream decoder, including tiled boundaries.
- Benchmarked paired versus fused RMS/RoPE at 1, 256, and 2,048 tokens. The fused operation was faster on this device and passed reference parity checks. Different BF16 kernels can change sampled songs; cross-backend bit-identical generation is not promised.
- The real-checkpoint upstream reference and adapted runtime each produced valid eight-second stereo clips.
- A real ComfyUI server accepted and executed all four custom nodes, core Preview Audio, and core Save Audio (Advanced).

| Queue test | Result |
|---|---|
| Full score + music + decode + preview/save | Complete 66.7187-second song, no truncation; approximately 56.9 seconds including HTTP polling on the first final-build run |
| Cached repeat | Successful; reused generated audio |
| Direct generation, CFG 1.01 | Successful eight-second capped clip; truncation reported |
| Supplied melody score | Successful eight-second capped clip; truncation reported |
| Cancellation during generation | ComfyUI reported `execution_interrupted` |
| Generation after cancellation | Successful new clip |
| Composition-only workflow | Successful ABC output through core Preview as Text; no music rendering |
| Combined workflow, eight-second audio test caps | All four branches passed on a fresh ComfyUI instance; three stereo FLACs, previews, and score text. Included the exact failing instrumental style with blank lyrics and full planning; Compose reported automatic direct generation. |
| Combined workflow, shipped duration settings | All four branches passed in 86.7 seconds: 66.72-second vocal song, 34.32-second supplied-score song, and 60-second capped instrumental. All saved files were finite, non-silent, 48 kHz stereo. Instrumental duration truncation was reported as expected. |
| Updated combined workflow with Score Editor | All four sections and all 19 nodes completed successfully in 124.9 seconds. The editor supplied its validated eight-bar instrumental score through a STRING connection to Compose, then Render/Decode/core Preview/Save produced a finite, non-silent 48 kHz stereo clip capped at 45 seconds. The cap was reported; structural score validity does not establish exact audio adherence. |
| Piano-roll conversion through generation | Rebuilt the score from piano-roll note/chord events, preserved parsed musical content, and executed Editor → Compose → Render → Decode → core Preview/Save. Produced a finite, non-silent 48 kHz stereo clip with a 16-second test cap in 14.5 seconds; `piano_editor_00001.flac`. |

The complete-song FLAC is stereo, 48,000 Hz, 3,202,496 samples/channel. Peak amplitude 0.80371; RMS 0.12073; all samples finite. An independent local Whisper transcription recovered all eight requested lyric lines, with one trailing extra word. This verifies intelligibility and lyric following for this example; it is not a comprehensive music-quality evaluation.

Local outputs and detailed queue records are in `ComfyUI/output/yue2_validation/`. The complete example is `full_song_00001.flac`; the original draft integration also produced a 58.8-second song in `output/audio/YuE2/`.

## Limits

Visual browser interaction could not be completed: the browser blocked local-page navigation, and automatic approval review rejected launching an isolated test browser with only “blocked by policy” as its reason. Backend node registration, workflow execution, preview-file creation, and saved audio were verified. Visual layout and widget interaction remain unverified.

For the Score Editor update, browser-harness reached Chrome's “Allow remote debugging?” permission prompt and stopped pending user approval. Visual layout, browser file dialogs and workflow reload interaction remain unverified. No permission bypass was attempted. Editor generation was tested on a separate local ComfyUI instance on port 8190.

The piano-roll update had no available browser-harness connection. Live visual layout, pointer interaction in an actual browser, and audible Web Audio preview remain unverified; the DOM fixture tests do not replace these checks.

The installed pack and its training nodes were loaded successfully in the main ComfyUI session on port 8188.

Peak VRAM was not instrumented. The server's post-generation Torch allocation was approximately 7.0 GiB; this is retained model memory, not a peak-memory claim. A 24 GB configuration, other GPU vendors, and multi-user concurrent execution were not tested.

Pinned SheetSage2 audio-to-score transcription is included for joint-training dataset preparation. Quantization, CUDA graphs, and the legacy benchmark decoder remain outside this release. Score-conditioned music generation is included.
