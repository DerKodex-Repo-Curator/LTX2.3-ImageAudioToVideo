"""
Single-stage Audio-to-Video pipeline optimized for MPS.

Avoids the two-stage pipeline's model reload + 2x upscaling that crashes on MPS.
Runs denoising at target resolution directly with one transformer load.
Models are loaded sequentially and freed after use to stay under 128GB.
"""

import gc
import logging
import os
import time
from collections.abc import Iterator

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.helpers import (
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import ModalitySpec

logger = logging.getLogger(__name__)


class A2VidOneStage:
    """
    Single-stage audio-to-video pipeline.
    Generates video at target resolution directly (no upsampling stage).
    Models are loaded one at a time and freed after use to manage memory.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        # Store config for lazy model loading in __call__
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._loras = loras
        self._quantization = quantization
        self._registry = registry

    def _log_memory(self, tag: str):
        """Log current memory usage for diagnostics."""
        parts = [f"[Memory] {tag}"]
        try:
            rss = os.getpid()
            import psutil
            proc = psutil.Process()
            rss_gb = proc.memory_info().rss / (1024 ** 3)
            parts.append(f"RSS={rss_gb:.1f}GB")
        except ImportError:
            pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                alloc_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
                parts.append(f"MPS_alloc={alloc_gb:.1f}GB")
            except Exception:
                pass
        elif torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            parts.append(f"CUDA_alloc={alloc_gb:.1f}GB")
        logger.info(" | ".join(parts))

    def _flush_memory(self, tag: str = ""):
        """Force garbage collection and clear device caches."""
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if tag:
            self._log_memory(f"after flush: {tag}")

    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        images: list,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        # Resolution must be multiples of 32
        height = (height // 32) * 32
        width = (width // 32) * 32

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        self._log_memory("pipeline start")

        # --- Encode prompts, then free text encoder (~24GB) ---
        logger.info("Loading prompt encoder...")
        t0 = time.time()
        prompt_encoder = PromptEncoder(
            self._checkpoint_path, self._gemma_root, self.dtype, self.device,
            registry=self._registry,
        )
        self._log_memory("prompt encoder loaded")

        logger.info("Encoding prompts...")
        ctx_p, ctx_n = prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        del prompt_encoder
        self._flush_memory("prompt encoder freed")
        logger.info(f"  Prompts encoded + encoder freed ({time.time()-t0:.0f}s)")

        # --- Encode audio, then free audio encoder (~4GB) ---
        logger.info("Loading audio conditioner...")
        t0 = time.time()
        audio_conditioner = AudioConditioner(
            self._checkpoint_path, self.dtype, self.device,
            registry=self._registry,
        )
        self._log_memory("audio conditioner loaded")

        logger.info("Encoding audio...")
        decoded_audio = decode_audio_from_file(
            audio_path, self.device, audio_start_time, audio_max_duration
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}")

        encoded_audio_latent = audio_conditioner(
            lambda enc: vae_encode_audio(decoded_audio, enc, None)
        )

        del audio_conditioner
        self._flush_memory("audio conditioner freed")
        logger.info(f"  Audio encoded + encoder freed ({time.time()-t0:.0f}s)")

        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16
        )
        target_frames = audio_shape.frames
        actual_frames = encoded_audio_latent.shape[2]
        if actual_frames > target_frames:
            encoded_audio_latent = encoded_audio_latent[:, :, :target_frames]
        elif actual_frames < target_frames:
            pad_size = target_frames - actual_frames
            encoded_audio_latent = torch.nn.functional.pad(
                encoded_audio_latent, (0, 0, 0, pad_size), mode="replicate"
            )
            logger.info(f"  Padded audio latent from {actual_frames} to {target_frames} frames")

        # --- Encode image conditionings, then free image encoder (~8GB) ---
        logger.info("Loading image conditioner...")
        t0 = time.time()
        image_conditioner = ImageConditioner(
            self._checkpoint_path, self.dtype, self.device,
            registry=self._registry,
        )
        self._log_memory("image conditioner loaded")

        logger.info("Encoding image conditioning...")
        conditionings = image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )

        del image_conditioner
        self._flush_memory("image conditioner freed")
        logger.info(f"  Image encoded + encoder freed ({time.time()-t0:.0f}s)")

        # --- Diffusion: load transformer (~44GB), denoise, then free ---
        logger.info("Loading diffusion transformer...")
        t0 = time.time()
        diffusion = DiffusionStage(
            self._checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(self._loras),
            quantization=self._quantization,
            registry=self._registry,
        )
        self._log_memory("diffusion transformer loaded")

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        logger.info(f"Denoising {num_inference_steps} steps at {width}x{height}...")
        video_state, _ = diffusion(
            denoiser=GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=MultiModalGuiderParams(),
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        del diffusion, sigmas, conditionings
        del v_context_p, v_context_n, a_context_p, encoded_audio_latent
        self._flush_memory("diffusion transformer freed")
        logger.info(f"  Denoising complete + transformer freed ({time.time()-t0:.0f}s)")

        # --- Decode video, then free decoder (~4GB) ---
        logger.info("Loading video decoder...")
        t0 = time.time()
        video_decoder = VideoDecoder(
            self._checkpoint_path, self.dtype, self.device,
            registry=self._registry,
        )
        self._log_memory("video decoder loaded")

        logger.info("Decoding video...")
        decoded_video = video_decoder(video_state.latent, tiling_config, generator)

        del video_decoder, video_state
        self._flush_memory("video decoder freed")
        logger.info(f"  Video decoded + decoder freed ({time.time()-t0:.0f}s)")

        # Return original audio for fidelity
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )

        self._log_memory("pipeline complete")
        return decoded_video, original_audio
