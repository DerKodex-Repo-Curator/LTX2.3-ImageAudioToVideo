# LTX-2.3 Image + Audio to Video

Generate lip-synced talking-head videos from a still portrait image and an audio file. Built on [Lightricks' LTX-2](https://github.com/Lightricks/LTX-2) v2.3 — a 22B-parameter DiT-based audio-video foundation model.

**Input:** portrait image + audio clip + text prompt **→ Output:** MP4 video with synchronized lip movements

## Features

- **Two-stage pipeline** — Stage 1 denoises at half resolution, spatial 2x upscale, stage 2 refines at full resolution with distilled LoRA
- **Automatic resolution & duration** — matches source image aspect ratio and audio length
- **Memory-efficient** — staged transformer loading with aggressive cleanup; runs on 128GB Apple Silicon
- **STG (Spatiotemporal Guidance)** — configurable temporal coherence enhancement
- **Single-stage fallback** — MPS-compatible alternative pipeline included

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/techfreakworm/LTX2.3-ImageAudioToVideo.git
cd LTX2.3-ImageAudioToVideo

# 2. Run setup (clones LTX-2, creates venv, installs dependencies)
bash setup.sh

# 3. Download models (see Models section below)

# 4. Generate
source .venv/bin/activate
python generate.py \
  --image inputs/portrait.png \
  --audio inputs/speech.wav \
  --prompt "A person speaking clearly with natural expressions"
```

## Models

Download and place in the `models/` directory:

| Model | Size | Download |
|-------|------|----------|
| LTX-2.3 Checkpoint | ~44GB | [ltx-2.3-22b-dev.safetensors](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors) |
| Distilled LoRA | ~1.5GB | [ltx-2.3-22b-distilled-lora-384.safetensors](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors) |
| Spatial Upscaler 2x | ~150MB | [ltx-2.3-spatial-upscaler-x2-1.1.safetensors](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors) |

**Gemma 3 Text Encoder** — download all files from [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) into `models/gemma_root/`:

```bash
# Using huggingface CLI
pip install huggingface_hub
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir models/gemma_root
```

## Usage

```bash
# Quality mode (default config — ~25 min, STG enabled)
python generate.py \
  --image inputs/portrait.png \
  --audio inputs/speech.wav \
  --prompt "A woman speaking enthusiastically with expressive gestures"

# Fast mode (~5 min, STG disabled)
python generate.py \
  --config config.fast.yaml \
  --image inputs/portrait.png \
  --audio inputs/speech.wav \
  --prompt "A person speaking clearly"

# With overrides
python generate.py \
  --image inputs/photo.png \
  --audio inputs/voice.wav \
  --prompt "A man explaining something" \
  --output outputs/my_video.mp4 \
  --seed 123 \
  --steps 15 \
  --fps 25
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image`, `-i` | required | Input portrait image |
| `--audio`, `-a` | required | Input audio file (stereo; mono auto-converted) |
| `--prompt`, `-p` | required | Text prompt describing the scene |
| `--output`, `-o` | `outputs/gen_<timestamp>.mp4` | Output video path |
| `--config`, `-c` | `config.yaml` | Config file path |
| `--seed`, `-s` | 42 | Random seed |
| `--num-frames` | auto | Frame count override (must be 8k+1) |
| `--height` | auto | Height override (snapped to 32-multiple) |
| `--width` | auto | Width override (snapped to 32-multiple) |
| `--steps` | from config | Stage 1 inference steps |
| `--fps` | 25.0 | Frame rate |
| `--negative-prompt` | from config | Negative prompt override |

## Architecture

### Two-Stage Pipeline (`a2vid_preloaded.py`)

1. **Init** — loads shared models (Gemma text encoder, VAE encoders, upsampler, decoder)
2. **Encode** — prompts via Gemma, audio via audio VAE, image via video encoder; each model freed after use
3. **Stage 1** — denoise at half resolution (N steps), then free transformer (~44GB reclaimed)
4. **Upsample** — 2x spatial latent upscale
5. **Stage 2** — refine at full resolution (4 distilled steps with LoRA), then free transformer
6. **Decode** — video VAE decode + mux original audio into MP4

Audio is "frozen" during diffusion — it conditions video generation but is not modified. The original waveform is muxed into the output for full fidelity.

### Single-Stage Fallback (`a2vid_one_stage.py`)

Denoises directly at target resolution in one pass. No upsampling. Use this if the two-stage pipeline causes memory issues on your hardware.

## Memory Management

Designed for machines with 128GB unified memory (Apple Silicon). Key optimizations:

- **Staged transformer loading** — only one 22B transformer in memory at a time
- **Aggressive model cleanup** — `model.to("meta")` immediately releases storage after each model's job is done
- **Resolution cap** — `max_pixels` in config prevents OOM at large resolutions (default: 409,600 = ~640x640)
- **Sequential pipeline** — text encoder, audio encoder, video encoder freed as soon as their outputs are captured

Peak memory: ~65-70GB (during transformer diffusion steps).

## Configuration

Two configs included:

- **`config.yaml`** — Quality mode (20 steps, STG on, ~25 min)
- **`config.fast.yaml`** — Fast mode (10 steps, STG off, ~5 min)

### Key Parameters

| Parameter | Quality | Fast | Effect |
|-----------|---------|------|--------|
| `num_inference_steps` | 20 | 10 | More steps = sharper details |
| `video_stg_scale` | 1.0 | 0.0 | STG improves temporal coherence |
| `video_stg_blocks` | [28] | [] | Which transformer block for STG |
| `video_cfg_scale` | 3.0 | 3.5 | Classifier-free guidance strength |
| `max_pixels` | 409600 | 409600 | Resolution safety cap |

## Prompting Tips

- Write detailed, chronological descriptions in a single flowing paragraph
- Start directly with the main action
- Include specific movements, gestures, and expressions
- Describe character appearance, background, lighting, camera angle
- Keep within 200 words

## Requirements

- Python 3.11+
- PyTorch with MPS (macOS) or CUDA support
- 128GB unified memory (Apple Silicon) recommended. May run on 64GB with very high memory pressure and reduced resolution — expect heavy swap usage and significantly slower generation
- ~50GB disk space for models

## License

This project wraps [Lightricks' LTX-2](https://github.com/Lightricks/LTX-2), which has its own license. See `LTX-2/LICENSE` after running setup.

## Acknowledgments

- [Lightricks](https://github.com/Lightricks/LTX-2) for the LTX-2 foundation model
- [Google](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) for the Gemma 3 text encoder
