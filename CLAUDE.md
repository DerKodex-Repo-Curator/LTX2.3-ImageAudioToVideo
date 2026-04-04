# LTX-2.3 Image + Audio to Video (Lip Sync)

## Project Overview

Generates lip-synced talking-head videos from a still portrait image and an audio file. Built on Lightricks' LTX-2 v2.3 — a 22B-parameter DiT-based audio-video foundation model. Takes an image, audio, and text prompt as input; outputs an MP4 where the person appears to speak the audio with synchronized lip movements.

## Architecture

Two pipeline implementations wrapping the upstream `ltx-core` and `ltx-pipelines` packages:

### Two-Stage Pipeline (`a2vid_preloaded.py`) — Primary

`A2VidPreloaded` loads shared models at init, then builds/frees transformers on-demand to stay under 128GB.

**Stage 1:** Denoise at half resolution (W/2 x H/2) for N steps using the base transformer.
**Spatial Upscale:** 2x latent upscaling via a dedicated spatial upsampler model.
**Stage 2:** Refine at full resolution for 4 fixed distilled steps (sigma values: `[0.909375, 0.725, 0.421875, 0.0]`) using the transformer with distilled LoRA applied.

Memory management order:
1. Load text encoder + embeddings processor → encode prompts → free (~24GB reclaimed)
2. Load audio encoder → encode audio → free
3. Build stage 1 transformer → run diffusion → `model.to("meta")` + free (~44GB reclaimed)
4. Upsample latent → free upsampler
5. Build stage 2 conditionings → free video encoder
6. Build stage 2 transformer → run diffusion → `model.to("meta")` + free
7. Decode video via video decoder

### Single-Stage Pipeline (`a2vid_one_stage.py`) — MPS Fallback

`A2VidOneStage` uses block-based loading from `ltx_pipelines.utils.blocks`. Denoises directly at target resolution in one pass — no upsampling. Exists because the two-stage pipeline's spatial upsampler can crash on MPS.

### Audio Handling

Audio is NOT generated — it conditions the video. The audio latent is encoded once via the audio VAE, then "frozen" during diffusion (denoise_mask set to zeros). The original audio waveform is muxed back into the output MP4.

The audio VAE requires **stereo input**. Mono files are auto-converted by `ensure_stereo()` in `generate.py`.

## File Structure

```
generate.py            # CLI entry point — parses args, loads config, runs pipeline
a2vid_preloaded.py     # Two-stage pipeline (staged transformer loading, memory-efficient)
a2vid_one_stage.py     # Single-stage pipeline (MPS-compatible fallback)
config.yaml            # Quality config (20 steps, STG on, ~25 min)
config.fast.yaml       # Fast config (10 steps, STG off, ~5 min)
setup.sh               # Automated setup: clone LTX-2, create venv, install deps
.gitignore             # Ignores .venv/, LTX-2/, models/, outputs/
inputs/                # Input files (user-provided images and audio)
outputs/               # Generated MP4 videos (timestamped filenames)
models/                # Downloaded model files (not committed)
  gemma_root/          # Gemma text encoder + tokenizer files
LTX-2/                 # Cloned Lightricks/LTX-2 repo (not committed)
  packages/ltx-core/   # Core model components (installed as editable package)
  packages/ltx-pipelines/ # Pipeline utilities (installed as editable package)
.venv/                 # Python 3.11 virtual environment (not committed)
```

## Setup

```bash
bash setup.sh
# Downloads/places models in models/ directory
# See README.md for model download links
```

## Key Constraints

### Memory (128GB Apple Silicon)
- Staged transformer loading: only one 22B transformer in memory at a time
- Models freed via `model.to("meta")` + `gc.collect()` + `torch.mps.synchronize()` + `torch.mps.empty_cache()`
- Resolution capped via `max_pixels` config (default 409,600) to prevent OOM at stage 2
- Peak memory: ~65-70GB during transformer diffusion steps

### Frame Count (8k+1 Format)
The transformer's temporal architecture requires `num_frames = 8k + 1` (e.g., 9, 17, 25, ..., 121, ...).

### Resolution
All dimensions are snapped to multiples of 32. Default behavior matches input image resolution (capped by `max_pixels`).

### Audio
- Audio VAE requires **stereo** input. Mono auto-converted.
- Audio latent is frozen during diffusion — conditions video but is not modified.

## Config

Two configs shipped:
- `config.yaml` — Quality mode: 20 steps, STG on (`stg_scale=1.0`, `stg_blocks=[28]`)
- `config.fast.yaml` — Fast mode: 10 steps, STG off (`stg_scale=0.0`, `stg_blocks=[]`)

## Development Notes

- `generate.py` is the only CLI entry point. Both pipeline classes are importable modules.
- Upstream `ltx-core` and `ltx-pipelines` are editable installs from `LTX-2/packages/` — changes there take effect immediately.
- No tests or CI exist. Validate changes by running generation end-to-end.
- Python 3.11 is used in the venv.
