# Future Improvements

Known limitations and planned fixes for the LTX-2.3 Image+Audio to Video pipeline, particularly on Apple Silicon (MPS).

---

## 1. Enable STG (Spatio-Temporal Guidance) on MPS

**Status:** Blocked — STG produces corrupted output on Apple Silicon MPS.

### Problem

With `video_stg_scale > 0`, the output becomes washed out with no lip sync or meaningful motion. On CUDA, STG=1.0 with `stg_blocks=[28]` works correctly and improves temporal coherence. On MPS, it destroys the output entirely.

### Root Cause (Theory)

STG computes a perturbation delta: `stg_scale * (conditioned_output - perturbed_output)`, where the perturbed pass skips self-attention in the specified transformer block. These two tensors are large and nearly equal — their difference is the meaningful signal.

In **bfloat16** (8-bit mantissa), subtracting two nearly-equal values causes **catastrophic cancellation**: the significant digits cancel and the result is dominated by rounding noise. On CUDA, the rounding behavior is well-tested and the difference stays meaningful. On MPS, Apple's GPU appears to produce slightly different rounding, making the subtraction capture noise instead of signal. This noise, multiplied by `stg_scale` and applied at every denoising step, compounds into corrupted output.

### Proposed Fix

**Cast the STG subtraction to float32** in `_guided_denoise()` at `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py`.

The fix would be in the `MultiModalGuider.calculate()` method at `LTX-2/packages/ltx-core/src/ltx_core/components/guiders.py`, approximately where the guidance formula is computed:

```python
# Current (bfloat16 — catastrophic cancellation on MPS):
stg_delta = self.params.stg_scale * (cond - perturbed)

# Proposed fix (float32 subtraction):
stg_delta = self.params.stg_scale * (cond.float() - perturbed.float()).to(cond.dtype)
```

### How to Test

1. Apply the float32 cast patch to the upstream code
2. Set `video_stg_scale: 1.0` and `video_stg_blocks: [28]` in `config.yaml`
3. Generate a video and verify:
   - Output is not washed out
   - Lip sync is present
   - Temporal coherence is improved compared to STG=0
4. If it works, submit the fix as a PR to Lightricks/LTX-2 with MPS test results

### Fallback

If float32 cast doesn't fix it, try:
- Lower `stg_scale` values (0.1, 0.2, 0.3) to reduce noise amplification
- Different `stg_blocks` values (try [27], [29])
- Run the entire perturbation pass in float32 (heavier but more reliable)

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
