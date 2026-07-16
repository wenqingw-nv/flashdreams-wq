<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Drift correction on FlashDreams HY-WorldPlay — results report (2026-07-16)

Counterfactual-Forcing drift corrector (frozen-base LoRA, 14.75M params / ~0.3%, **zero real videos
end-to-end**) ported to the HY-WorldPlay WAN-5B integration (distilled 4-step, camera/action-conditioned
world model). ~2.5 GPU-days on one shared GB300. Corrector merges into the weights at deploy → **zero
inference overhead**.

## T1 — Closed-loop drift & quality (headline)

Full-resolution seed, 3 unseen camera trajectories, ~19s rollouts (24 chunks), matched seeds.
Δ-drift = MUSIQ(first 20%) − MUSIQ(all), lower is better (BAgger's metric). Port success bar: ≥30% Δ cut.

| config | MUSIQ ↑ | MUSIQ late ↑ | Δ-drift ↓ | Δ reduction |
|---|---|---|---|---|
| HY-WorldPlay base | 69.1 | 69.6 | +0.60 | — |
| + corrector v2 (gain 1.0) | 71.1 | 69.9 | +0.30 | **−50%** |
| + corrector v2 (gain 0.85) | 71.3 | 69.8 | +0.26 | **−58%** |
| + corrector v2, α*(t)-gated | **73.5** | **73.1** | **−1.43 (negative)** | **> −100%** |

The gated configuration reaches the *negative-drift* regime (quality rises over the rollout) that the
reference method achieved on its paper host (av2: −0.16), at +4.4 MUSIQ over base. Qualitatively, the
base's characteristic late-horizon failure (saturation/contrast runaway, texture mush) is eliminated;
structures (bridge, pavilion) stay intact. Side-by-side videos: ``outputs/eval_v2bridge/sbs_*.mp4``.

## T2 — Open-loop corrector fit (method validity)

Val R² = fraction of the drift-induced x̂0 error removed, held-out scenes (37 clips); kill gate was 0.15.

| checkpoint | val R² |
|---|---|
| v1 (counterfactual teacher) | 0.453 |
| v2 (+DAgger, +contraction) | 0.549 |
| v3 (+round-2 DAgger) | 0.556 |

## T3 — Step-0 diagnostic (α*, method transfer finding)

Faithful drift pairs, M=8 seeds, unbiased estimator. The drift-induced error is ~95% systematic on the
reference host at every timestep; on this distilled host it is timestep-dependent:

| t | 1000 | 960 | 888 | 727 |
|---|---|---|---|---|
| α* (systematic fraction) | **0.81** | 0.53 | 0.53 | 0.58 |

First host where α*(t) is non-flat → the Drift-SNR gate becomes a live deploy knob (T1's best row)
rather than a diagnostic. Cross-host: correction gains track α* as the theory predicts.

## Caveats (open items, fix designed and scheduled)

1. **Progression bias (primary open item):** corrected rollouts under-advance along the camera
   trajectory (sim-to-start 0.62 vs base 0.42; reference-host scale: SF 0.17, av2s 0.50). Root causes
   identified cross-host (loop-structured training pairs + the contraction term); the fix (mixed-geometry
   pairs + trust-region term, contraction off) is implemented and on hold pending the paper-host fix
   landing first (owner priority call, ``OWNER_FEEDBACK.md``). An earlier "2.4× revisit-memory
   improvement" claim is **retracted** — inflated by this same bias.
2. **Motion level:** RAFT dynamic degree −30..40% vs base at high gain (progressive; same root cause).
3. **Saturation:** v2 overshoots saturation upward (sat-drift 0.070 vs base 0.045, single-cell) — the
   v1↔v2 DAgger oscillation; round-2 (v3) targets it, eval pending the hold.
4. 18-scene held-out finals excluded: low-res seed frames broke the Δ-MUSIQ anchor (protocol artifact,
   not a method result). Short-horizon do-no-harm batch not yet run.

## Metric definitions

- **Δ-drift**: MUSIQ(first 20% of frames) − MUSIQ(all frames), per-frame at chunk stride.
- **sim-to-start**: mean last-20% frame-correlation to the first-2s mean frame (progression probe;
  reference-repo convention).
- **dynamic degree**: mean RAFT flow magnitude between scored frames (motion-freeze guard).
- **sharpness ratio**: late/early variance-of-Laplacian (compounding-blur guard).
- **sat-drift**: mean |saturation − saturation(frame 0)|.
- **val R²**: 1 − ‖x̂0_corr − x̂0_clean‖²/‖x̂0_clean − x̂0_base‖² on held-out drift pairs.

## Deploy recommendation (draft, pending v4)

If deploying today: **v2 @ α*(t) gate** for quality/drift-critical use where reduced camera progression
is acceptable, or **v2 @ 0.7** as the guard-compliant conservative point (Δ −26%, dynamics within 20%
of base). Recommended path: wait for v4 (mixed-geometry + trust region, contraction off) — designed to
restore progression at ≥ these quality numbers — then merge the winning LoRA into the checkpoint
(zero-overhead) behind a runner flag.

## Story in one paragraph (for slides)

The drift corrector ported cleanly to a production-class world model with zero real data: a measured
step-0 diagnostic predicted reduced-but-real headroom, training hit the expected open-loop fit
(R² 0.45–0.56), and the deployed corrector cut the drift metric 50–58% — reaching negative drift with
per-step gating — while eliminating the base's visible saturation-runaway failure. The port also
surfaced a genuine, now well-characterized limitation (progression bias from reference-distribution
inheritance) that reproduces on the paper host, has a designed fix, and turned into a cross-host
finding: correction strength should follow the measured per-timestep systematicity α*(t).

## Artifacts

Checkpoints ``outputs/lora_{v1,v2,v3}.pt`` · scores ``outputs/eval*/scores.json`` · videos/strips
``outputs/eval_v2bridge/`` · gate ``outputs/gate/gate_faithful.json`` · decisions ``OWNER_FEEDBACK.md`` ·
open items ``TODO.md``. Branch ``hy-worldplay-counterfactual-forcing`` (pushed to
``wenqingw-nv/flashdreams-wq``).
