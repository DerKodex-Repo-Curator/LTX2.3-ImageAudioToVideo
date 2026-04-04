"""
Single-stage Audio-to-Video pipeline optimized for MPS.

Avoids the two-stage pipeline's model reload + 2x upscaling that crashes on MPS.
Runs denoising at target resolution directly with one transformer load.
"""

import logging
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
    Loads the transformer only once. Much more MPS-friendly.
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

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root, self.dtype, self.device, registry=registry
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.diffusion = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry
        )

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

        # Encode prompts
        logger.info("Encoding prompts...")
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding

        # Encode audio
        logger.info("Encoding audio...")
        decoded_audio = decode_audio_from_file(
            audio_path, self.device, audio_start_time, audio_max_duration
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}")

        encoded_audio_latent = self.audio_conditioner(
            lambda enc: vae_encode_audio(decoded_audio, enc, None)
        )
        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16
        )
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]

        # Encode image conditionings at target resolution
        logger.info("Encoding image conditioning...")
        conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )

        # Single-stage denoising at target resolution
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        logger.info(f"Denoising {num_inference_steps} steps at {width}x{height}...")
        video_state, _ = self.diffusion(
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

        # Decode video
        logger.info("Decoding video...")
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)

        # Return original audio for fidelity
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )

        return decoded_video, original_audio
