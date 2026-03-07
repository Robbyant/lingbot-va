# Acceleration Ideas for World-Action Models (WAM)

In this project, inference cost is dominated by:
- **Video token count** (often much larger than action tokens; e.g., Robotwin is ~30:1).
- **Video diffusion steps** (many backbone calls per chunk).

Below are algorithm-level acceleration directions, especially for optimizing **video latent** inference.

## A) Fewer steps (distillation / consistency / rectified flow)
- Distill the video branch from many steps (e.g., 25) to **1–4 steps** using teacher trajectories.
- `FlowMatchScheduler.step()` is Euler-like integration, suitable for progressive distillation / consistency training.
- Keep action head unchanged initially; aggressively compress **video steps** first.

## B) Fewer video tokens (structural latent compression)
1. **Low-res latent diffusion + latent super-resolution**
   - Diffuse on smaller `(H', W')` latent grids (token count \(\propto\) area), then use a lightweight decoder/upsampler to recover full latent size.
2. **Learned bottleneck tokens (per-frame K summary tokens)**
   - Encode each frame latent grid into **K \(\ll\) H×W** tokens; diffuse only these tokens; decode back only if needed.
3. **Camera/ROI-aware compression**
   - Keep high-res tokens only for critical views/regions (e.g., wrist / end-effector neighborhood), downsample the rest.

## C) Fewer tokens without changing output (dynamic token selection)
- Update only a subset of video tokens per step; reuse previous values for the rest.
- Token importance can be estimated by motion/changes, end-effector proximity, or early-step attention statistics.
- Combine with `flex_attention` masks to realize sparse compute beyond fixed windows.

## D) Fewer backbone calls (share computation between video and action)
- Instead of running separate diffusion loops for video then action, predict actions from shared hidden states during the video loop.
- Alternatively reduce action diffusion to 1–2 steps, or switch action to deterministic regression + uncertainty head.

## E) More reuse across chunks (beyond KV cache)
- Extend from KV cache to **state/token cache**: cache a compressed world-state representation across chunks and only update the increment.
- Use a **keyframe strategy**: refresh video latent features at low frequency; run action at high frequency; periodic correction.

## F) Fast-path inference (no explicit video generation)
- If deployment only needs control success (not visualization), do not generate full video latents at inference time.
- Train with video loss to maintain representation quality, but deploy a fast-path that outputs only compact features needed for action.

## Recommended MVP routes
- **Route 1 (most reliable):** video step distillation (25 → 4 → 2 → 1).
- **Route 2 (token bottleneck):** latent token compression (low-res or K-summary) + lightweight decoding.
- **Route 3 (research-y):** dynamic sparse attention + keyframe refresh policy.

