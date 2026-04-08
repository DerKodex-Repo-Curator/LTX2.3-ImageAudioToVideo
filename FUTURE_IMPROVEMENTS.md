# Future Improvements

Known limitations and planned fixes for the LTX-2.3 Image+Audio to Video pipeline, particularly on Apple Silicon (MPS).

---

## 1. Enable STG (Spatio-Temporal Guidance) on MPS

**Status:** Blocked — STG produces corrupted output on Apple Silicon MPS.

### Problem

With `video_stg_scale > 0`, the output becomes washed out with no lip sync or meaningful motion. On CUDA, STG=1.0 with `stg_blocks=[28]` works correctly and improves temporal coherence. On MPS, it destroys the output entirely.

### Root Cause (Researched)

Three confirmed MPS precision issues compound to corrupt STG output:

**1. MPS "fast math" ([pytorch/pytorch#84936](https://github.com/pytorch/pytorch/issues/84936)):**
Apple's MPSGraph compiles ALL Metal kernels with fast math enabled. Operations like `exp()`, `softmax()`, and attention produce ~1e-7 different results per operation vs CUDA. There is NO PyTorch setting to disable this. Apple acknowledged but never fixed it.

**2. bfloat16 attention masking errors ([pytorch/pytorch#163588](https://github.com/pytorch/pytorch/issues/163588)):**
Flex Attention with bfloat16 produces up to 3.4% relative error when masking positions. STG masks self-attention in block 28 for the perturbation pass — directly affected.

**3. Error amplification chain:**
The `cond` and `ptb` passes take different paths through block 28 (real attention vs V-only bypass). Over 30 transformer blocks of cross-attention + feedforward, MPS fast-math errors (~1e-7 per op) accumulate differently in each pass. The subtraction `cond - ptb` then REVEALS these accumulated errors instead of capturing the intended STG signal. The rescaling (`cond.std() / pred.std()` in bfloat16) further amplifies the noise.

**Key precision loss points** (from code trace):
- `guiders.py:259` — STG delta: `stg_scale * (cond - ptb)` in bfloat16
- `guiders.py:264-265` — Rescale: `cond.std() / pred.std()` in bfloat16
- `attention.py:234-235` — Perturbation mask blend: `out * mask + v * (1-mask)` in bfloat16
- `batch_split.py:75-85` — If cond and ptb land in different chunks, MPS non-determinism adds further divergence

### Fix Strategy: Approach A then Approach C

#### Approach A: Float32 guidance calculation (try first — zero overhead)

Cast the entire guidance formula and rescaling to float32 in `MultiModalGuider.calculate()` at `ltx-core/components/guiders.py`:

```python
def calculate(self, cond, uncond_text, uncond_perturbed, uncond_modality):
    # Upcast to float32 for precision-sensitive guidance math
    dtype = cond.dtype
    cond, uncond_text, uncond_perturbed, uncond_modality = (
        t.float() if isinstance(t, torch.Tensor) else t
        for t in (cond, uncond_text, uncond_perturbed, uncond_modality)
    )
    pred = (
        cond
        + (self.params.cfg_scale - 1) * (cond - uncond_text)
        + self.params.stg_scale * (cond - uncond_perturbed)
        + (self.params.modality_scale - 1) * (cond - uncond_modality)
    )
    if self.params.rescale_scale != 0:
        factor = cond.std() / pred.std()
        factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
        pred = pred * factor
    return pred.to(dtype)
```

**Likelihood of success:** ~30%. Fixes the subtraction and rescaling precision but not the accumulated MPS fast-math errors inside the transformer.

#### Approach C: Separate float32 forward pass for perturbation (escalation — heavy but reliable)

If Approach A fails, run the STG perturbation as a completely separate transformer forward pass in float32:

1. In `_guided_denoise()` at `ltx-pipelines/utils/denoisers.py`, split the perturbation pass out of the batch
2. Run cond+uncond+mod as one batched bfloat16 pass (normal)
3. Run ptb alone as a separate float32 forward pass
4. Cast cond to float32, compute `cond - ptb` in float32, cast back

This eliminates ALL MPS fast-math errors for the ptb pass. Cost: one extra full transformer forward pass in float32 (~30-50% slower per step).

**Likelihood of success:** ~90%.

### How to Test

1. Apply the fix to the LTX-2 submodule
2. Set `video_stg_scale: 1.0` and `video_stg_blocks: [28]` in `config.yaml`
3. Generate with the working pirate portrait + short audio (< 6s)
4. Verify output is not washed out, has lip sync, and shows improved temporal coherence vs STG=0

---

## 2. Increase Steps and Resolution on MPS

**Status:** Needs investigation.

### Problem

- **Steps > 10** break lip sync: the video has motion but lips don't move to match audio
- **max_pixels > 409600** (~512x768) produces garbage output at any step count
- Both work fine on CUDA (upstream defaults are 30 steps at much higher resolution)

### Current Workaround

Cap at 10 steps and 409600 max_pixels. Use `--upscale 2` (ffmpeg lanczos) for higher output resolution.

### Investigation Plan

1. **Steps issue**: Test intermediate values (12, 15) to find the exact threshold. Check if the `LTX2Scheduler`'s token-dependent sigma shifting produces a qualitatively different schedule at different step counts on MPS. Log and compare sigma values between 10-step and 20-step runs.

2. **Resolution issue**: Test resolutions incrementally (409600 → 450000 → 500000 → 550000 → 614400) to find the exact threshold. Check if the issue is the Stage 1 half-resolution, the spatial upsampler at larger sizes, or the Stage 2 refinement at full resolution.

3. **Precision hypothesis**: If both issues share a root cause (bfloat16 precision on MPS), a targeted float32 cast at critical computation points might fix both. The denoising loop's Euler step update (`x = x + dt * dx`) is another candidate for precision loss over many iterations.

---

## 3. AI-based Video Upscaling

**Status:** Enhancement — current ffmpeg lanczos upscaling works but could be better.

### Current State

The `--upscale 2` flag uses ffmpeg's lanczos filter for 2x/4x upscaling. Fast and preserves lip sync perfectly, but the result is a scaled-up version of the low-res output — no detail hallucination.

### Planned Improvements

1. **Real-ESRGAN**: Frame-by-frame AI upscaling. The `realesrgan-ncnn-vulkan` binary supports macOS. Can also be converted to CoreML for Neural Engine acceleration on Apple Silicon (up to 78x speedup vs CPU PyTorch).

2. **Video-specific super-resolution**: Models like BasicVSR++ or RealBasicVSR that exploit temporal information across frames for more temporally consistent upscaling.

3. **Frame interpolation**: Models like RIFE or FILM for 2x frame rate (25fps → 50fps), making motion smoother without re-generating.

### Implementation

Add `--upscale-model` flag to `generate.py`:
```bash
python generate.py --image ... --audio ... --prompt "..." --upscale 2 --upscale-model realesrgan
```

Fallback to ffmpeg lanczos if the AI model isn't installed.
