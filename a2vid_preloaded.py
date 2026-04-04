"""
Two-stage Audio-to-Video pipeline with staged transformer loading.

Loads shared models (text encoder, VAE encoders/decoders, upsampler) once at
init. Transformers are loaded one at a time — stage 1 transformer is built,
used, and freed before stage 2 transformer (with distilled LoRA) is loaded.
This keeps peak memory under 128GB on Apple Silicon.
"""

import gc
import logging
import time
from collections.abc import Iterator
from typing import TypeVar

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, SDOps
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    AudioEncoderConfigurator,
    encode_audio as vae_encode_audio,
)
from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator, X0Model
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
    get_video_chunks_number,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    combined_image_conditionings,
    create_noised_state,
    get_device,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec

from dataclasses import replace

logger = logging.getLogger(__name__)
_M = TypeVar("_M", bound=torch.nn.Module)


def _build_state(spec, tools, noiser, dtype, device):
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


class A2VidPreloaded:
    """
    Two-stage audio-to-video with staged transformer loading.
    Loads shared models at init. Transformers are loaded/freed one at a time
    during __call__ to keep peak memory under 128GB.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora_path: str,
        distilled_lora_strength: float,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._loras = loras or []
        self._checkpoint_path = checkpoint_path
        self._distilled_lora_path = distilled_lora_path
        self._distilled_lora_strength = distilled_lora_strength
        self._quantization = quantization
        self._registry = DummyRegistry()

        t0 = time.time()

        # 1. Build Gemma text encoder
        logger.info("Loading Gemma text encoder...")
        module_ops = module_ops_from_gemma_root(gemma_root)
        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

        self.text_encoder = (
            Builder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=self._registry,
            )
            .build(device=self.device, dtype=self.dtype)
            .eval()
        )
        logger.info(f"  Text encoder loaded ({time.time()-t0:.0f}s)")

        # 2. Build embeddings processor
        t1 = time.time()
        logger.info("Loading embeddings processor...")
        self.embeddings_processor = (
            Builder(
                model_path=checkpoint_path,
                model_class_configurator=EmbeddingsProcessorConfigurator,
                model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
                registry=self._registry,
            )
            .build(device=self.device, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"  Embeddings processor loaded ({time.time()-t1:.0f}s)")

        # 3. Build video encoder (for image conditioning)
        t1 = time.time()
        logger.info("Loading video encoder...")
        self.video_encoder = (
            Builder(
                model_path=checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self._registry,
            )
            .build(device=self.device, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"  Video encoder loaded ({time.time()-t1:.0f}s)")

        # 4. Build audio encoder
        t1 = time.time()
        logger.info("Loading audio encoder...")
        self.audio_encoder = (
            Builder(
                model_path=checkpoint_path,
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self._registry,
            )
            .build(device=self.device, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"  Audio encoder loaded ({time.time()-t1:.0f}s)")

        # 5. Build spatial upsampler
        t1 = time.time()
        logger.info("Loading spatial upsampler...")
        self.upsampler = (
            Builder(
                model_path=spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self._registry,
            )
            .build(device=self.device, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"  Upsampler loaded ({time.time()-t1:.0f}s)")

        # 6. Build video decoder
        t1 = time.time()
        logger.info("Loading video decoder...")
        self.video_decoder = (
            Builder(
                model_path=checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self._registry,
            )
            .build(device=self.device, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"  Video decoder loaded ({time.time()-t1:.0f}s)")

        # Transformers are NOT loaded here — they are built on-demand in __call__
        logger.info(f"Shared models loaded in {time.time()-t0:.0f}s (transformers loaded on-demand)")

    def _build_transformer_s1(self):
        """Build stage 1 transformer (no distilled LoRA)."""
        logger.info("Loading transformer (stage 1)...")
        t0 = time.time()
        builder = Builder(
            model_path=self._checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(self._loras),
            registry=self._registry,
        )
        if self._quantization is not None:
            builder = builder.with_module_ops(
                (*builder.module_ops, *self._quantization.module_ops)
            ).with_sd_ops(
                SDOps(
                    name=f"chain_{builder.model_sd_ops.name}+{self._quantization.sd_ops.name}",
                    mapping=(*builder.model_sd_ops.mapping, *self._quantization.sd_ops.mapping),
                )
            )
        transformer = X0Model(
            builder.build(device=self.device)
        ).to(self.device).eval()
        logger.info(f"  Transformer stage 1 loaded ({time.time()-t0:.0f}s)")
        return transformer

    def _build_transformer_s2(self):
        """Build stage 2 transformer (with distilled LoRA)."""
        logger.info("Loading transformer (stage 2 + distilled LoRA)...")
        t0 = time.time()
        distilled_lora_spec = LoraPathStrengthAndSDOps(
            self._distilled_lora_path, self._distilled_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP
        )
        loras = (*tuple(self._loras), distilled_lora_spec)
        builder = Builder(
            model_path=self._checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=loras,
            registry=self._registry,
            lora_load_device=self.device,
        )
        if self._quantization is not None:
            builder = builder.with_module_ops(
                (*builder.module_ops, *self._quantization.module_ops)
            ).with_sd_ops(
                SDOps(
                    name=f"chain_{builder.model_sd_ops.name}+{self._quantization.sd_ops.name}",
                    mapping=(*builder.model_sd_ops.mapping, *self._quantization.sd_ops.mapping),
                )
            )
        transformer = X0Model(
            builder.build(device=self.device)
        ).to(self.device).eval()
        logger.info(f"  Transformer stage 2 loaded ({time.time()-t0:.0f}s)")
        return transformer

    def _flush_memory(self):
        """Force garbage collection and clear device caches."""
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _release_model(self, model: torch.nn.Module):
        """Move model to meta device to immediately release all storage."""
        model.to("meta")

    def _free_models(self, *names: str):
        """Release model storage and delete attributes by name."""
        for name in names:
            model = getattr(self, name, None)
            if model is not None and isinstance(model, torch.nn.Module):
                model.to("meta")
            if hasattr(self, name):
                delattr(self, name)
        self._flush_memory()

    def _encode_prompts(self, prompts: list[str]):
        raw_outputs = [self.text_encoder.encode(p) for p in prompts]
        return [
            self.embeddings_processor.process_hidden_states(hs, mask)
            for hs, mask in raw_outputs
        ]

    def _run_diffusion(self, transformer, denoiser, sigmas, noiser, width, height,
                       frames, fps, video_spec, audio_spec):
        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)

        v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        v_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
        v_state = _build_state(video_spec, v_tools, noiser, self.dtype, self.device)

        a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        a_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
        a_state = _build_state(audio_spec, a_tools, noiser, self.dtype, self.device)

        v_state, a_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=v_state,
            audio_state=a_state,
            stepper=EulerDiffusionStep(),
            transformer=transformer,
            denoiser=denoiser,
        )

        v_state = v_tools.clear_conditioning(v_state)
        v_state = v_tools.unpatchify(v_state)
        if a_state is not None:
            a_state = a_tools.clear_conditioning(a_state)
            a_state = a_tools.unpatchify(a_state)

        return v_state, a_state

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
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        # Resolution must be multiples of 32
        height = (height // 32) * 32
        width = (width // 32) * 32

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        # --- Encode prompts, then free text encoder (~24GB) ---
        logger.info("Encoding prompts...")
        ctx_p, ctx_n = self._encode_prompts([prompt, negative_prompt])
        v_ctx_p, a_ctx_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_ctx_n, _ = ctx_n.video_encoding, ctx_n.audio_encoding
        self._free_models("text_encoder", "embeddings_processor")
        logger.info("  Text encoder freed.")

        # --- Encode audio, then free audio encoder ---
        logger.info("Encoding audio...")
        decoded_audio = decode_audio_from_file(
            audio_path, self.device, audio_start_time, audio_max_duration
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}")
        encoded_audio = vae_encode_audio(decoded_audio, self.audio_encoder, None)
        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16
        )
        # Clip or pad audio latent to match target video duration
        target_frames = audio_shape.frames
        actual_frames = encoded_audio.shape[2]
        if actual_frames > target_frames:
            encoded_audio = encoded_audio[:, :, :target_frames]
        elif actual_frames < target_frames:
            pad_size = target_frames - actual_frames
            encoded_audio = torch.nn.functional.pad(encoded_audio, (0, 0, 0, pad_size), mode="replicate")
            logger.info(f"  Padded audio latent from {actual_frames} to {target_frames} frames")
        self._free_models("audio_encoder")
        logger.info("  Audio encoder freed.")

        # --- Stage 1: denoise at half resolution ---
        s1_h, s1_w = height // 2, width // 2
        logger.info(f"Stage 1: denoising {num_inference_steps} steps at {s1_w}x{s1_h}...")

        s1_cond = combined_image_conditionings(
            images=images, height=s1_h, width=s1_w,
            video_encoder=self.video_encoder, dtype=self.dtype, device=self.device,
        )

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        transformer_s1 = self._build_transformer_s1()

        v_state, _ = self._run_diffusion(
            transformer=transformer_s1,
            denoiser=GuidedDenoiser(
                v_context=v_ctx_p, a_context=a_ctx_p,
                video_guider=MultiModalGuider(params=video_guider_params, negative_context=v_ctx_n),
                audio_guider=MultiModalGuider(params=MultiModalGuiderParams()),
            ),
            sigmas=sigmas, noiser=noiser,
            width=s1_w, height=s1_h, frames=num_frames, fps=frame_rate,
            video_spec=ModalitySpec(context=v_ctx_p, conditionings=s1_cond),
            audio_spec=ModalitySpec(context=a_ctx_p, frozen=True, noise_scale=0.0,
                                   initial_latent=encoded_audio),
        )

        transformer_s1.to("meta")
        del transformer_s1, s1_cond, sigmas
        self._flush_memory()
        logger.info("Stage 1 complete — transformer freed.")

        # --- Upsample 2x, then free upsampler ---
        logger.info("Upsampling latent 2x...")
        upscaled = upsample_video(
            latent=v_state.latent[:1],
            video_encoder=self.video_encoder,
            upsampler=self.upsampler,
        )
        del v_state
        self._free_models("upsampler")
        logger.info("  Upsampler freed.")

        # --- Build stage 2 conditionings, then free video encoder ---
        s2_cond = combined_image_conditionings(
            images=images, height=height, width=width,
            video_encoder=self.video_encoder, dtype=self.dtype, device=self.device,
        )
        self._free_models("video_encoder")
        logger.info("  Video encoder freed.")

        # --- Stage 2: refine at full resolution ---
        logger.info(f"Stage 2: refining 4 steps at {width}x{height}...")
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        transformer_s2 = self._build_transformer_s2()

        v_state, _ = self._run_diffusion(
            transformer=transformer_s2,
            denoiser=SimpleDenoiser(v_ctx_p, a_ctx_p),
            sigmas=distilled_sigmas, noiser=noiser,
            width=width, height=height, frames=num_frames, fps=frame_rate,
            video_spec=ModalitySpec(context=v_ctx_p, conditionings=s2_cond,
                                   noise_scale=distilled_sigmas[0].item(),
                                   initial_latent=upscaled),
            audio_spec=ModalitySpec(context=a_ctx_p, frozen=True, noise_scale=0.0,
                                   initial_latent=encoded_audio),
        )

        transformer_s2.to("meta")
        del transformer_s2, s2_cond, upscaled, distilled_sigmas
        del v_ctx_p, v_ctx_n, a_ctx_p, encoded_audio
        self._flush_memory()
        logger.info("Stage 2 complete — transformer freed.")

        # --- Decode video ---
        logger.info("Decoding video...")
        decoded_video = self.video_decoder.decode_video(
            v_state.latent, tiling_config, generator
        )

        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )

        return decoded_video, original_audio
