<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Counterfactual-Forcing port — status & TODO

Port of the drift-corrector LoRA (Counterfactual Forcing, v2 = DAgger + drift-contraction) onto the
HY-WorldPlay integration. Zero-real-data regime: strafe-loop rollouts with lap-1 clean teachers, first
frames synthesized from MovieGen prompts. Checkpoints/evals under `outputs/` (gitignored); decisions and
metrics logged in `OWNER_FEEDBACK.md` and `outputs/*/scores.json`.

## Resolved

| issue | solution |
|---|---|
| Step-0 gate below reference bar (α* 0.53–0.81 vs Wan2.1's 0.91–0.99) | Proceeded per owner; α*(t) is **non-flat** on this host → gate became a deploy knob (`TGATE=1`) |
| Trainer grads silently severed (rolling-KV write path) | `_train_attn.py` functional dual-branch attention (verified ≈ stock, mean |Δx̂0| 0.01) |
| In-place inference-only Triton RoPE breaks autograd | Differentiable torch RoPE matching the kernel's lane layout (pair k ← lane 2k) |
| ~40 GiB tape / OOM next to co-tenant jobs | Per-block checkpointing + two-phase backward (dot-product chain rule), single cache; VAE CUDA-graphs off |
| v1 blur + structure loss (bridge fade) | v2 DAgger + contraction: structure preserved, MUSIQ 71.1 vs base 69.1, Δ-drift −50% (bridge cells) |
| Metrics gameable one-by-one | Scorer carries MUSIQ Δ, RAFT dynamics, sharpness ratio, sat-drift, advance-score + strips |

## Open — fixes queued (v4 chain running, final iteration)

| issue | solution | status |
|---|---|---|
| **Repeat prior**: loop-trained corrector doesn't follow forward trajectories (advance-score 0.38 vs base 0.58; stuck/repeats, hallucinated objects); gate/gain can't fix (in weights) | Mixed-geometry pairs (legs 3/4/5 break the fixed revisit lag) + trust-region `FID_W ∈ {0.3,0.1,1.0}` (drift-gap-normalized, warmed up) | v4 arms queued |
| **Contraction stickiness**: progressive motion decay (dyn 20→9); owner cross-host: contraction kills progression even without loops | Contraction OFF in v4; v3 (CW .25) vs v3b (CW 0) gives this host's ablation row | v3/v3b tonight |
| **Saturation overshoot** (v2 sat-drift 0.070 vs base 0.045; v1↔v2 DAgger oscillation) | Round-2 DAgger pool (v3+) + verify in grid; stat-matching term only if it persists | in grid |
| Revisit-memory "2.4×" claim inflated by repeat prior | Retracted; re-report only for checkpoints with advance-score ≈ base | in writeup |

## Backlog (post-v4 / optional)

- Full-res held-out seed frames (480×832 upscales broke Δ-MUSIQ on the 18-scene finals; single-file
  Wan2.2 ckpt 404 → sharded index works, decode OOM at 704×1280 needs chunked decode or more headroom).
- Official VBench 6-dim + sat-drift suite via the on-box Self-Forcing harness (paper-comparable table).
- Grad-through-prefill in the dagger term (corrector currently can't reshape memory *encoding*).
- q/k-only LoRA arm (attention routing without content-writing v/o); must train from scratch.
- LoRA→weights merge for zero-overhead deploy + runner flag, if a config passes.
- 50s-horizon eval; `corr05` half-gain anomaly unexplained (likely meaningless interpolated LoRA state).
- Longer-term: 2nd-host caveats for the paper (host-dependent gains, non-flat α*(t), Δ-metric assumption).

## Decision rule (owner, `OWNER_FEEDBACK.md`)

v4 pass bar = advance-score ≈ base (0.58) at ≥ baseline quality plus standing guards (Δ ≥30% cut, dyn,
sat, sharpness). Pass → deploy config + merge. Fail → deploy-config recommendation from the grid + failure
analysis (loop-teacher bias, contraction stickiness) as the final deliverable.
