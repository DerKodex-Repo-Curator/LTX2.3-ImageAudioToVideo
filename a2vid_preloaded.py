"""
Two-stage Audio-to-Video pipeline using upstream DiffusionStage blocks.

Based on upstream a2vid_two_stage.py. Stage 1 denoises at half resolution with
audio frozen, Stage 2 upsamples and refines at full resolution with distilled LoRA.
All models lazy-load in __call__ and free after use (managed by block classes).
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
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.types import ModalitySpec

logger = logging.getLogger(__name__)


class A2VidPreloaded:
    """
    Two-stage audio-to-video using upstream DiffusionStage blocks.
    All models lazy-load in __call__ and are freed after use by the block classes.
    Peak memory: one 22B transformer (~44GB) + block overhead at a time.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora_path: str,
        distilled_lora_strength: float,
        spatial_upsampler_path: str,
        gemma_root: str,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        distilled_lora_spec = LoraPathStrengthAndSDOps(
            distilled_lora_path, distilled_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP
        )

        # Block classes store config only — no weights loaded until __call__
        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root, self.dtype, self.device,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device,
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path, self.dtype, self.device,
        )
        self.stage_1 = DiffusionStage(
            checkpoint_path, self.dtype, self.device,
            loras=(),
            quantization=quantization,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path, self.dtype, self.device,
            loras=(distilled_lora_spec,),
            quantization=quantization,
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path, self.dtype, self.device,
        )

    def _log_memory(self, tag: str):
        """Log current memory usage for diagnostics."""
        parts = [f"[Memory] {tag}"]
        try:
            import psutil
            rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
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
        audio_guider_params: MultiModalGuiderParams,
        images: list,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        # Resolution must be multiples of 32
        height = (height // 32) * 32
        width = (width // 32) * 32

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        self._log_memory("pipeline start")

        # --- 1. Encode prompts ---
        logger.info("Encoding prompts...")
        t0 = time.time()
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        self._flush_memory("prompts encoded")
        logger.info(f"  Prompts encoded ({time.time()-t0:.0f}s)")

        # --- 2. Encode audio ---
        logger.info("Encoding audio...")
        t0 = time.time()
        decoded_audio = decode_audio_from_file(
            audio_path, self.device, audio_start_time, audio_max_duration
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}")

        encoded_audio_latent = self.audio_conditioner(
            lambda enc: vae_encode_audio(decoded_audio, enc, None)
        )
        # Clip or pad audio latent to match target video duration
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

        self._flush_memory("audio encoded")
        logger.info(f"  Audio encoded ({time.time()-t0:.0f}s)")

        # --- 3. Stage 1: denoise at half resolution, audio frozen ---
        s1_h, s1_w = height // 2, width // 2
        logger.info(f"Stage 1: denoising {num_inference_steps} steps at {s1_w}x{s1_h}...")
        t0 = time.time()

        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images, height=s1_h, width=s1_w,
                video_encoder=enc, dtype=self.dtype, device=self.device,
            )
        )

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        video_state, _ = self.stage_1(
            denoiser=GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=s1_w,
            height=s1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
        )

        self._flush_memory("stage 1 complete")
        logger.info(f"  Stage 1 complete ({time.time()-t0:.0f}s)")

        # --- 4. Upsample 2x ---
        # MPS can crash on einops rearrange + depthwise conv in upsampler.
        # Fall back to CPU if that happens (~150MB model, <60s on CPU).
        logger.info("Upsampling latent 2x...")
        t0 = time.time()
        try:
            upscaled_video_latent = self.upsampler(video_state.latent[:1])
        except RuntimeError as e:
            if "mps" not in str(e).lower() and "not supported" not in str(e).lower():
                raise
            logger.warning(f"MPS upsampler failed ({e}), falling back to CPU...")
            from ltx_core.model.upsampler import upsample_video
            # Build encoder and upsampler on CPU manually
            cpu = torch.device("cpu")
            encoder = self.upsampler._encoder_builder.build(device=cpu, dtype=self.dtype).to(cpu).eval()
            up_model = self.upsampler._upsampler_builder.build(device=cpu, dtype=self.dtype).to(cpu).eval()
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1].to(cpu),
                video_encoder=encoder,
                upsampler=up_model,
            ).to(self.device)
            del encoder, up_model

        del video_state
        self._flush_memory("upsampled")
        logger.info(f"  Upsampled ({time.time()-t0:.0f}s)")

        # --- 5. Stage 2: refine at full resolution, audio frozen ---
        logger.info(f"Stage 2: refining 4 steps at {width}x{height}...")
        t0 = time.time()

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images, height=height, width=width,
                video_encoder=enc, dtype=self.dtype, device=self.device,
            )
        )

        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=distilled_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
        )

        del upscaled_video_latent, distilled_sigmas
        self._flush_memory("stage 2 complete")
        logger.info(f"  Stage 2 complete ({time.time()-t0:.0f}s)")

        # --- 6. Decode video ---
        logger.info("Decoding video...")
        t0 = time.time()
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)

        # Return original audio waveform for fidelity (not VAE-decoded)
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )

        self._log_memory("pipeline complete")
        logger.info(f"  Video decoded ({time.time()-t0:.0f}s)")

        return decoded_video, original_audio
