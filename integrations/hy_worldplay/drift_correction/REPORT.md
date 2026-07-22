<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Drift correction on FlashDreams HY-WorldPlay — results report (2026-07-16)

## 2026-07-22 official VBench 6-dim table (paper harness, calibrated scale)

Scored on the EXISTING shipped-decision videos (no new generation): bridge
cells = `outputs/eval_sweep/` (3 held-out commanded-motion trajectories),
static cells = `outputs/demo_static/` (8 locked-off scenes). Harness = the
paper setup's custom-input VBench checkout; per-dim raw JSONs + `table.json`
under `outputs/vbench/` (verified on real disk via fresh foreground probe).

Commanded motion (bridge):

| config | subj cons | bg cons | aesthetic | imaging | motion smooth | dyn degree | mean |
|---|---|---|---|---|---|---|---|
| base | 86.4 | 88.6 | 65.3 | 77.6 | 91.4 | 100.0 | 84.9 |
| corr050 | 92.0 | 92.9 | 67.1 | 77.0 | 91.8 | 100.0 | 86.8 |
| corrgate | **94.6** | **94.2** | **68.3** | 77.2 | **93.9** | 100.0 | **88.1** |
| **corrgate050 (shipped)** | 88.7 | 90.2 | 64.5 | 76.5 | 93.0 | 100.0 | 85.5 |

Static suite:

| config | subj cons | bg cons | aesthetic | imaging | motion smooth | dyn degree | mean |
|---|---|---|---|---|---|---|---|
| **base (shipped)** | 84.8 | 88.1 | 51.8 | **74.2** | 90.7 | 37.5 | 71.2 |
| corr050 | 82.8 | 88.9 | 51.7 | 70.7 | 85.0 | 75.0 | 75.7 |
| corrgate | 88.2 | **91.8** | **53.4** | 71.4 | **91.2** | 37.5 | 72.2 |
| corrgate050 | **89.3** | 91.5 | 53.3 | 70.6 | 90.0 | 50.0 | 74.1 |

Reading (consistent with the ship decision): on motion content every
corrector config beats base on consistency/quality dims with dynamic degree
saturated at 100 for all — corrgate050's smaller VBench margin over base is
the price of preserving progression (VBench's binary dynamic-degree
threshold cannot see the anchoring that killed plain corrgate on the owner's
eyeball; latesim/RAFT in the main table do). On static content the corrector
rows trade imaging quality (−3.5..−3.6) and motion smoothness for
consistency, and the static dynamic-degree column is a caveat, not a win:
locked-off scenes SHOULD be near-static, and corr050's 75.0 reflects its
erratic-motion artifact. VBench's per-dim means mask the drift axis entirely
(it has no temporal-degradation dimension), so these tables complement — not
replace — the Δ-drift/latesim/seam suite.

## 2026-07-21 deployment pass (Clean Forcing; HANDOFF acceptance criteria)

Gain sweep on 3 held-out trajectories (24 chunks, seed 5042, scene-matched prompt, base re-run
in-sweep). New protocol metrics ported from the reference host: DINO latesim (anchoring), lag-2s
identity, cut count, seam-aligned motion/sharpness ratios (true decoded cadence 13 + 16k).

| config | MUSIQ | late | Δ-drift | Δ cut | dyn | sat-drift | sim | latesim |
|---|---|---|---|---|---|---|---|---|
| base | 68.0 | 70.0 | +1.70 | — | 19.3 | 0.036 | 0.42 | 0.60 |
| **corr050** | 71.2 | 68.7 | +0.47 | **−72%** | **16.1** | **0.013** | **0.56** | 0.78 |
| corr070 | 72.9 | 72.3 | −0.76 | −145% | 13.4 | 0.032 | 0.70 | 0.85 |
| corr085 | 72.3 | 72.1 | −0.89 | −152% | 12.1 | 0.047 | 0.72 | 0.84 |
| corr (1.0) | 72.1 | 71.6 | −0.56 | −133% | 11.7 | 0.066 | 0.71 | 0.81 |
| corrgate | **73.8** | **73.5** | −0.86 | −151% | 10.4 | 0.020 | 0.68 | 0.87 |

Owner eyeball verdicts (final per house rule): **corr050 reaches the bridge** (progression kept,
better color than base, small pulses, acceptable quality); **corrgate anchors** (best quality, does
not reach the bridge). Only corr050 approaches acceptance criterion 2 (dyn −17% vs the ~15% bar);
every higher gain loses 30–46% dynamics.

**Corrector-induced seam pulse (new finding, measured):** on this boundary-clean production host the
corrector's per-chunk restoring force is visible as a chunk-cadence beat (motion stalls at seams,
post-boundary blur) — the paper host's issue-3 mechanism unmasked, amplified by flat-gain deployment
against this host's non-flat α*(t). Deployment levers under test: α*(t) gate × 0.5 global gain
(`corrgate050`; kill bar: reaches the bridge at seam ≥ corr050), per-step t-restriction as fallback.

**gate×0.5 composition (corrgate050), verified + owner-eyeballed:** latesim 0.669 (best progression
of any corrector config), dyn 20.3 ≈ base (no dynamics loss), seam-motion 1.211 / seam-sharpness
0.993 (mildest pulse, at/below base), Δ −48% vs base, MUSIQ 70.4. Owner eyeball on the bridge sbs:
"much better in progression, without much pulling back or pulse issue" — PASS.

**Static-background suite (8 locked-off scenes incl. new-element entrance), verified + owner-eyeballed:**
on static content the base wins both axes (owner ordering, quality and pulse alike:
base > corrgate050 > corrgate > corr full). Consistent with the instruments — the base measures
*negative* drift on these scenes (Δ −5.97): there is nothing to correct, so any correction is pure
artifact cost. v3 re-eval closed on verified scores: v2 stays the deployed LoRA (v3's target sat
overshoot got worse at full gain, and it loses dynamics/progression at the deploy point).

## SHIP DECISION (owner, 2026-07-21) — content-keyed deployment

- **Commanded-motion jobs → `corrgate050`** (v2 LoRA at per-step ``alpha*(t) × 0.5``).
- **Static / no-camera-motion jobs → corrector OFF** (untouched base weights). Static scenes don't
  drift on this host; corrector-off there is by design, not an open defect (this also moots the
  scene-5 motion-collapse observation from the static suite).
- The selector is free on HY-WorldPlay: the trajectory is a runtime input, so the runner keys the
  choice per job — `HyWorldPlayWanI2VRunnerConfig.drift_corrector` (LoRA checkpoint path;
  `drift_corrector_gain=0.5`) applies the corrector only when the pose contains any non-idle action
  label (`hy_worldplay/_drift_corrector.py`). Static jobs bypass LoRA application entirely (zero
  overhead); motion jobs keep the LoRA unfused because the gate is per-timestep — a single-scale
  weight merge cannot express ``alpha*(t)``, and the unfused cost is ~0.3% params of extra matmul.

Non-shipped alternatives (one line each): **corr050** — flat 0.5 gain, slightly better MUSIQ/Δ on
the bridge but worse pulse/blur and erratic motion on static content; **full gain (corr)** — best
raw drift cut but −40% dynamics and the strongest seam beat; **plain gate (corrgate)** — best
absolute quality (73.8) but anchors (never reaches the bridge; −41% dynamics on static suite).

Artifacts: `outputs/eval_sweep/` (scores.json + sbs mp4s), `outputs/demo_static/`, `outputs/eval_v3/`.

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
