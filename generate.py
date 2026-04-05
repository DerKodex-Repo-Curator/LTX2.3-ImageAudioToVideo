#!/usr/bin/env python3
"""
LTX-2.3 Image + Audio to Video with Lip Sync

Two-stage pipeline (default): half-res denoise → learned 2x upsample → full-res refine.
Single-stage fallback: denoises at target resolution directly.
Matches source image resolution and audio duration automatically.
"""

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.resolve()


def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_DIR / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("models", {}).items():
        if isinstance(val, str):
            cfg["models"][key] = str(Path(val).expanduser())
    return cfg


def resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return str(path.resolve())


def ensure_stereo(audio_path: str) -> str:
    """LTX-2.3 audio VAE requires stereo input."""
    import soundfile as sf
    import numpy as np

    info = sf.info(audio_path)
    if info.channels >= 2:
        return audio_path

    logger.info(f"Converting mono audio to stereo: {audio_path}")
    data, sr = sf.read(audio_path)
    data_stereo = np.stack([data, data], axis=-1)
    stereo_path = str(Path(audio_path).with_suffix(".stereo.wav"))
    sf.write(stereo_path, data_stereo, sr)
    return stereo_path


def get_image_dims(image_path: str, max_pixels: int = 0) -> tuple[int, int]:
    """Get image dimensions, snapped to multiples of 32.
    If max_pixels > 0, scale down to fit within that pixel budget while
    preserving aspect ratio (both dims stay multiples of 32).
    """
    from PIL import Image
    img = Image.open(image_path)
    w = (img.width // 32) * 32
    h = (img.height // 32) * 32
    if max_pixels > 0 and w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w = (int(w * scale) // 32) * 32
        h = (int(h * scale) // 32) * 32
    return w, h


def get_audio_duration(audio_path: str) -> float:
    import soundfile as sf
    return sf.info(audio_path).duration


def frames_for_duration(duration: float, fps: float) -> int:
    """Calculate num_frames (8k+1 format) to cover audio duration."""
    raw = int(math.ceil(duration * fps))
    k = math.ceil((raw - 1) / 8)
    return 8 * k + 1


def _log_system_memory():
    """Log system memory at startup to help diagnose OOM issues."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)
        logger.info(f"System memory: {total_gb:.0f}GB total, {used_gb:.0f}GB used, {avail_gb:.0f}GB available")
        if avail_gb < 70:
            logger.warning(
                f"Low memory: only {avail_gb:.0f}GB available. "
                f"LTX-2.3 needs ~50-70GB free. Consider closing other apps or using config.fast.yaml."
            )
    except ImportError:
        logger.info("Install psutil for memory diagnostics: pip install psutil")


def _build_two_stage(models: dict, device: torch.device):
    """Build the two-stage pipeline (faster + better quality)."""
    from a2vid_preloaded import A2VidPreloaded
    return A2VidPreloaded(
        checkpoint_path=resolve_path(models["checkpoint"]),
        distilled_lora_path=resolve_path(models["distilled_lora"]),
        distilled_lora_strength=models.get("distilled_lora_strength", 0.8),
        spatial_upsampler_path=resolve_path(models["spatial_upscaler"]),
        gemma_root=resolve_path(models["gemma_root"]),
        device=device,
    )


def _build_single_stage(models: dict, device: torch.device):
    """Build the single-stage pipeline (MPS fallback)."""
    from a2vid_one_stage import A2VidOneStage
    return A2VidOneStage(
        checkpoint_path=resolve_path(models["checkpoint"]),
        gemma_root=resolve_path(models["gemma_root"]),
        loras=[],
        device=device,
    )


@torch.inference_mode()
def generate(
    image_path: str,
    audio_path: str,
    prompt: str,
    output_path: str,
    cfg: dict,
    pipeline_mode: str = "auto",
    seed: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int | None = None,
    frame_rate: float | None = None,
    negative_prompt: str | None = None,
    image_strength: float | None = None,
) -> str:
    _log_system_memory()
    audio_path = ensure_stereo(audio_path)

    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    sys.path.insert(0, str(PROJECT_DIR))

    models = cfg["models"]
    gen = cfg["generation"]
    guide = cfg["guidance"]
    audio_cfg = cfg["audio"]
    perf = cfg.get("performance", {})

    _fps = frame_rate or gen["frame_rate"]
    _seed = seed if seed is not None and seed >= 0 else gen.get("seed", 42)
    _steps = num_inference_steps or gen["num_inference_steps"]
    _neg = negative_prompt or gen["negative_prompt"]

    # Match source image resolution (with optional pixel budget cap)
    max_px = perf.get("max_pixels", 0)
    if width is None or height is None:
        img_w, img_h = get_image_dims(image_path, max_pixels=max_px)
        _width = width or img_w
        _height = height or img_h
    else:
        _width = (width // 32) * 32
        _height = (height // 32) * 32

    # Match audio duration
    if num_frames is None:
        audio_dur = get_audio_duration(audio_path)
        _num_frames = frames_for_duration(audio_dur, _fps)
        logger.info(f"Audio duration: {audio_dur:.2f}s -> {_num_frames} frames ({_num_frames/_fps:.2f}s)")
    else:
        _num_frames = num_frames

    duration = _num_frames / _fps
    logger.info(f"Output: {_width}x{_height}, {_num_frames} frames @ {_fps}fps = {duration:.1f}s")
    logger.info(f"Image: {image_path}")
    logger.info(f"Audio: {audio_path}")
    
    # Auto-inject camera stabilizers if the user didn't specify cinematic motions
    if not any(k in prompt.lower() for k in ["tripod", "camera", "pan", "zoom"]):
        prompt = f"{prompt.strip()} Shot on a tripod, natural motion blur, locked camera stability, 4k."
        
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Seed: {_seed}, Steps: {_steps}")

    device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build pipeline
    use_two_stage = pipeline_mode in ("auto", "two-stage")
    t0 = time.time()
    if use_two_stage:
        logger.info("Pipeline: two-stage (half-res denoise → 2x upsample → full-res refine)")
        pipeline = _build_two_stage(models, device)
    else:
        logger.info("Pipeline: single-stage (direct full-res denoise)")
        pipeline = _build_single_stage(models, device)
    load_time = time.time() - t0

    from ltx_core.components.guiders import MultiModalGuiderParams
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=guide["video_cfg_scale"],
        stg_scale=guide["video_stg_scale"],
        rescale_scale=guide["video_rescale_scale"],
        modality_scale=guide["a2v_guidance_scale"],
        skip_step=guide["video_skip_step"],
        stg_blocks=guide["video_stg_blocks"],
    )
    logger.info(
        f"Guidance: video(cfg={guide['video_cfg_scale']} stg={guide['video_stg_scale']} "
        f"rescale={guide['video_rescale_scale']} modality={guide['a2v_guidance_scale']} "
        f"stg_blocks={guide['video_stg_blocks']}) audio(defaults, frozen)"
    )

    # Image conditioning strength: 1.0 freezes frame 0 (no lip movement).
    # 0.7-0.8 preserves identity while allowing lip sync (per ComfyUI workflows).
    _img_strength = image_strength if image_strength is not None else gen.get("image_conditioning_strength", 0.7)
    logger.info(f"Image conditioning strength: {_img_strength}")
    images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=_img_strength)]

    audio_start = audio_cfg.get("start_time", 0.0)
    audio_max_dur = audio_cfg.get("max_duration") or (_num_frames / _fps)

    tiling_config = TilingConfig.default()

    logger.info("Starting generation...")
    t1 = time.time()

    if use_two_stage:
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=_neg,
            seed=_seed,
            height=_height,
            width=_width,
            num_frames=_num_frames,
            frame_rate=_fps,
            num_inference_steps=_steps,
            video_guider_params=video_guider_params,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start,
            audio_max_duration=audio_max_dur,
            tiling_config=tiling_config,
        )
    else:
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=_neg,
            seed=_seed,
            height=_height,
            width=_width,
            num_frames=_num_frames,
            frame_rate=_fps,
            num_inference_steps=_steps,
            video_guider_params=video_guider_params,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start,
            audio_max_duration=audio_max_dur,
            tiling_config=tiling_config,
            enhance_prompt=perf.get("enhance_prompt", False),
            streaming_prefetch_count=perf.get("streaming_prefetch_count"),
            max_batch_size=perf.get("max_batch_size", 1),
        )

    gen_time = time.time() - t1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    video_chunks = get_video_chunks_number(_num_frames, tiling_config)

    logger.info(f"Encoding video to {output_path}...")
    encode_video(
        video=video,
        fps=_fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks,
    )

    total_time = time.time() - t0
    logger.info(f"Done! {output_path}")
    logger.info(f"Model load: {load_time:.0f}s | Generation: {gen_time:.0f}s | Total: {total_time:.0f}s")
    logger.info(f"Output: {output_path} | {_width}x{_height} @ {_fps}fps | {_num_frames} frames ({_num_frames/_fps:.1f}s) | seed={_seed} | total={total_time:.0f}s")
    return output_path


def upscale_video(input_path: str, scale: int = 2) -> str:
    """Upscale video using ffmpeg lanczos filter, preserving audio."""
    import subprocess

    output_path = input_path.replace(".mp4", f"_upscaled_{scale}x.mp4")
    logger.info(f"Upscaling {scale}x: {input_path} -> {output_path}")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale=iw*{scale}:ih*{scale}:flags=lanczos",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg upscale failed: {result.stderr}")
        return input_path

    logger.info(f"Upscaled to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Image+Audio to Video with Lip Sync")
    parser.add_argument("--image", "-i", required=True, help="Input portrait image")
    parser.add_argument("--audio", "-a", required=True, help="Input audio file")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--config", "-c", default=None, help="Config file")
    parser.add_argument("--pipeline", choices=["auto", "two-stage", "single-stage"],
                        default="auto", help="Pipeline: auto (two-stage), two-stage, or single-stage")
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None, help="Override frame count (8k+1)")
    parser.add_argument("--height", type=int, default=None, help="Override height")
    parser.add_argument("--width", type=int, default=None, help="Override width")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--image-strength", type=float, default=None,
                        help="Image conditioning strength (0.0-1.0). Lower=more motion/lip sync, higher=more identity. Default 0.3")
    parser.add_argument("--upscale", type=int, default=None, choices=[2, 4],
                        help="Upscale output video by 2x or 4x using ffmpeg lanczos after generation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    image_path = str(Path(args.image).resolve())
    audio_path = str(Path(args.audio).resolve())

    output_path = args.output or str(
        PROJECT_DIR / "outputs" / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    output_path = str(Path(output_path).resolve())

    result_path = generate(
        image_path=image_path,
        audio_path=audio_path,
        prompt=args.prompt,
        output_path=output_path,
        cfg=cfg,
        pipeline_mode=args.pipeline,
        seed=args.seed,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        frame_rate=args.fps,
        negative_prompt=args.negative_prompt,
        image_strength=args.image_strength,
    )

    if args.upscale:
        upscale_video(result_path, args.upscale)


if __name__ == "__main__":
    main()
