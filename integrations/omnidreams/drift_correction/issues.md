<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams port — quality issues board

**OWNER VERDICT (2026-07-23, eyeball on the sweep sbs): corrgate050 with v2@1000
(`outputs/lora_v2.pt` final, `eval_sweep_v2f`) is best overall — better than v2@600
(`eval_sweep_v2e`) and v1 (`eval_sweep`). This OVERRULES the instruments, which prefer v2@600
(corrgate050 Δ −1.34 vs v2@1000's −0.45; owner eyeball outranks instruments per house rules).**
Deploy recommendation: **PROVISIONAL — v2@1000 merged at α*(t)×0.5 (corrgate050)**; owner chose
RETRAIN (pairs-v3, issue #1 fix) before ship — recommendation stands only if the retrain does not
supersede it. Rules as on the HY board: gate first with pre-set kill bars · owner eyeball outranks
instruments · failed arms live in git history, not on this board.

| # | issue | latest fix | status | running now | ETA |
|---|---|---|---|---|---|
| 1 | **Repeat-prior artifact (owner class b)** — in `eval_sweep_v2f/...corrgate050_RIGHT_scen6_s5042.mp4`: missing zebra-crossing lines, repeated identical trucks/trailers lining both roadsides, repeated buildings. **Verified 2026-07-23, evidence FOR the repeat-prior mechanism** (training pairs tiled a FIXED 40-frame HDMap lap ×16, `build_pairs.py` — same failure class as HY's v2 repeat prior, fixed there by mixed LEGS lap lengths): (i) scen6 HDMap carries explicit crosswalk polygons + only sparse vehicle boxes at f280–640; base renders the zebra crossing and varied roadside buildings, corrgate050 suppresses the crossing and hallucinates a continuous band of identical trailers on both sides; (ii) roadside-strip temporal autocorrelation (late window, f280+): corrected self-similarity elevated at ALL lags (0.47–0.51 vs base 0.31–0.39 — novel-content suppression), with detrended bumps specifically at the 40/80/120-frame lap harmonics (mean +0.0037 at 40-multiples vs −0.0002 at non-multiples; NOT at generic 8-frame chunk multiples −0.0003; base flat ≈0). Evidence frames: `outputs/eval_sweep_v2f/repeat_prior_evidence/` | **pairs-v3 retrain DONE (2026-07-23)**: mixed tile lengths (32/40/48-frame laps rotated per clip, `LAP_MIX=4,5,6`) break the fixed revisit period. v1-v3 val R² +0.348 (bar +0.15 PASS) · v2-v3 val dag-R² +0.428 final / +0.484 val-peak@300 (PASS). **Artifacts FIXED at corrgate050 (both ckpts)**: scen6 zebra crossing rendered again (f160/f180), repeated-trailer wall gone (f320/480/600 follow the HDMap boxes); roadside self-similarity 0.350 (v2: 0.502, base: 0.382), 40-lag comb bump −0.001 (v2: +0.0037). Evidence: `outputs/eval_sweep_v3/repeat_prior_recheck/` | **RESOLVED pending owner eyeball** — sbs under `outputs/eval_sweep_v3{,p}/`; ship-vs-v2 trade-off: v3 removes the repeat prior but instruments give back some quality/motion vs v2@1000 (see #2 and the v3 sweep row below) | owner eyeball | done |
| 2 | **Fine-organic-detail blur + semantic slide (owner class a)** — occasional object blur and semantic drift on fine organic detail (blurry trees; leaves going full→bare) vs base | pairs-v3 did NOT improve it — the compounding-blur guard regressed: sharpness_ratio at corrgate050 = 0.612 (v3-final) / 0.572 (v3-valpeak) vs 0.757 (v2@1000) vs 0.869 (base); scen6 0.725 vs v2 0.792. The repeat prior masked blur behind copied crisp content; removing it exposes the corrector's conservative pull | OPEN — needs its own arm (candidates: lower gain on low-t steps, sharpness-aware gate, shorter v2-v3 training); do NOT stack onto #1 without a gate | — | on owner call |

## pairs-v3 sweep summary (2026-07-23, 4 scenes: collapse scen0 / train-val scen5 / held-out scen6-7)

At the deploy dial corrgate050 (α*(t)×0.5), scene-mean:

| config | Δ-drift | MUSIQ | dyn | latesim | seam | sharp |
|---|---|---|---|---|---|---|
| base | +2.44 | 49.1 | 16.6 | 0.808 | 1.043 | 0.869 |
| v2@1000 (`eval_sweep_v2f`, owner-preferred, repeat-prior artifacts) | **−0.45** | **50.7** | **13.4** | 0.818 | 1.141 | 0.757 |
| v3-final (`eval_sweep_v3`, artifact-free) | +0.24 | 47.6 | 10.7 | 0.826 | 1.168 | 0.612 |
| v3-valpeak@300 (`eval_sweep_v3p`, artifact-free) | +0.29 | 48.4 | 12.0 | 0.819 | 1.147 | 0.572 |

v3 keeps ~90% of the Δ-drift cut and eliminates both scen6 conditioning-fidelity artifacts, but gives
back quality/motion/sharpness vs v2@1000. Within v3, val-peak@300 ≥ final on MUSIQ/dyn/seam (sharp worse).

**OWNER EYEBALL VERDICT on eval_sweep_v3 (2026-07-24): corrgate050 = best of our arms but FAILS
overall** — scen0 blurry trees + repeated road scenes (same street view along the drive); scen6 blurry
trees, trees gradually disappear entirely, roadside morphed, same-street-view repetition; scen7 leaves
disappear and grow back (pulse). Owner eyeball outranks the instruments. Reading: the generic
self-copying caveat from the issue-#1 verification IS the live artifact — mixed lap lengths removed the
*period* shortcut, not the *loop prior*; every training pair still revisits its own content.

Queued fixes (owner 2026-07-24):
1. **KILLED (2026-07-24, both bars failed at both σ)** — inference-time low-pass correction
   (`LP_SIGMA` in `eval_rollouts.py`, v2-v3 val-peak @ corrgate050, scenes 0/5/6/7, 2x-forward
   test dial): sharpness 0.597 (σ1) / 0.643 (σ2) vs bar ≥ 0.757 (base 0.869) · Δ-drift cut
   retained 44% (σ1) / 71% (σ2) vs bar ≥ ~90% (Δ 1.50/0.92 vs noLP 0.29, base 2.44). Explicit
   checks: scen6 trees persist slightly better under σ2 (scen6 sharp 0.76 ≈ base) but stay
   stylized; scen7 leaf disappearance persists (both corrected arms bare by f400, base leafy);
   repeated-street-view persists (expected, not counted). Reading: the corrective signal and the
   blur/semantic-slide damage are NOT frequency-separable in the delta — blurring removes drift
   correction faster than it restores sharpness. Artifacts: `outputs/eval_lp{1,2}_v3p/` (scores,
   sbs, `eval_lp2_v3p/lp_recheck/` frame stacks). Sweep-side monotonicity note: σ2 beat σ1 on BOTH
   axes (single-seed noise or gate interaction — do not extrapolate to σ>2 without a fresh bar).
1a. **INTERIM SHIP (owner eyeball PASS 2026-07-24): `lora_v2_v3_valpeak.pt` @ α*(t)×0.25
   (corrgate025)** — owner verdict on `eval_g025_v3_valpeak`: best of all arms so far; better
   trees/foliage detail and consistency, less repeated street view. The instrument sharpness bar
   (0.598 < 0.80) is OVERRULED by the eyeball (5th eyeball-over-instrument case on this project;
   owner wrote "corrgate50" but the dir's only arm is corrgate025 — recorded as corrgate025).
   Residual artifact ON RECORD: scen6 still shows some repeated rows of buildings/shops and 3
   repeated white trucks — this is the pre-stated explicit check for the v4 sweep, and corrgate025
   is now the reference dial the v4 sweep must beat (corrgate025 is in the v4 dial grid).
1b. **corrgate025 gentle dial (owner fallback arm, 2026-07-24)** — v2-v3 val-peak at α*(t)×0.25
   (`eval_g025_v3_valpeak`; final ckpt run pending): Δ +0.99 (bar ≤ +1.5 PASS; 60% cut) · MUSIQ 50.1 >
   base · dyn 13.8 (−17%) · **sharpness 0.598 vs bar ≥ 0.80 FAIL** (scene-mean; driven by scen7 0.48 vs
   base 1.16 and scen5 0.45). BUT the explicit visual checks PASS: scen7 trees keep their leaves through
   f560 (vs bare at ×0.5), scen6 foliage/palms persist with varied roadside (`g025_recheck/` stacks).
   NOT auto-flagged as interim ship (numeric bar missed); owner eyeball may overrule — sbs under
   `outputs/eval_g025_v3_valpeak/`. Repeated-street-view: still present (training-side, not counted).
   Final-ckpt variant (`eval_g025_v3_final`): Δ +2.49 ≈ base (no drift cut) · sharp 0.552 — strictly
   worse; **val-peak is the only corrgate025 candidate**. Gate v4 side-note: the gentle dial's Δ cut is
   checkpoint-sensitive; val-peak@300 generalizes, step-1000 does not.
2. **HOLD lifted — owner GO 2026-07-24 — pairs-v4, non-looping conditioning (RUNNING)**: stitch distinct HDMap segments per
   combined round, both changes behind flags (`build_pairs_v4.py`, `PAIR_SCHEME=fork`, `UW=1`):
   (a) conditioning = each clip's OWN continuous 645-frame HDMap (the samples ship ~80 s) — no tiling,
   no stitching, no teleports, content never recurs. Clean-counterpart design decision (pre-stated):
   **re-anchored forks** — every 20 chunks fork a fresh image-anchored rollout seeded by re-encoding the
   drifted rollout's own decoded frame at the fork, rolled over the same upcoming HDMap window
   (5-frame anchor chunk covers the tail of the previous chunk, so fork chunk 1+m consumes exactly
   chunk c_s+m's conditioning — no lag). Rationale: the only content-matched cleaner-content source
   without revisits; targets the structure/sharpness drift share; known limitation: the anchor inherits
   low-frequency color drift, so that share is absent from the target. Gate bar (pre-stated, BEFORE
   training): unbiased α*(t=1000) ≥ 0.4 AND rel_v(t=1000) ≥ 0.15 (pairs-v2 reference 0.96/0.45) — fail
   ⇒ STOP, fork design dead. (b) uncertainty-weighted loss: per-token α* weights from 2 noise draws
   (`w = bias²/(bias²+var)`, weights on the DAgger term; contraction unweighted by design — it penalizes
   accumulation of whatever residual remains while the weights keep unpredictable content out of the
   target; `UW=0` = ablation arm). Chain: pairs → gate → v1-v4 (R² ≥ 0.15) → DAgger → v2-v4 (dag-R² ≥
   0.15, step-tagged + val-peak) → dial sweeps (final + val-peak, grid now incl. corrgate025) → report.

Checkpoints: `outputs/lora_v1_v3.pt`, `outputs/lora_v2_v3{,_valpeak,_stepN}.pt`; sbs (sides in
filename) under `outputs/eval_sweep_v3{,p}/`.

## pairs-v4 sweep results (2026-07-24, scenes 0/5/6/7, both ckpts; sbs under `eval_sweep_v4{,p}/`)

| arm (scene-mean) | Δ-drift | MUSIQ | dyn | latesim | sharp |
|---|---|---|---|---|---|
| base | +2.44 | 49.1 | 16.6 | 0.808 | 0.869 |
| v2-v3 vp @025 (interim ship) | +0.99 | 50.1 | 13.8 | 0.797 | 0.598 |
| **v4-valpeak @025** | **+1.26** | **48.7** | 12.5 | 0.778 | **0.94 (> base — first arm ever)** |
| v4-final @025 | +1.88 | 47.9 | 13.4 | 0.779 | 0.805 |
| v4-valpeak @050 | +9.16 | 37.1 | 9.7 | 0.658 | 0.414 |

The fork+UW recipe fixed the blur axis (sharpness 0.94 > base; scen7 trees keep leaves through f560,
no pulse — `eval_sweep_v4p/v4_recheck/` stacks) at a smaller Δ cut than the interim ship (48% vs 60%).
Explicit scen6 checks: crosswalk rendered; NO repeated building/shop rows; the f600 trailer row is
conditioning-consistent (base shows it too); residual: roadside self-similarity 0.433 vs base 0.382
(v2 was 0.502; v3 0.374) and slightly stippled canopies — owner eyeball to weigh. corrgate050
catastrophically overshoots on v4 (Δ +9.2, MUSIQ 37) — the working dial is ×0.25.
**Deploy recommendation: PROVISIONAL — v4-valpeak @ α*(t)×0.25 as the new candidate vs the interim
ship (blur fixed, drift cut slightly smaller); owner eyeball decides.**

Prospective-prediction outcome (see analysis below, recorded pre-sweep): top-1 HIT (corrgate025
predicted first by 0.098 vs 0.103) but the predicted near-tie with corrgate050 was badly wrong (050
collapsed) — the α*×ρ magnitude is right at the optimum but the metric is too flat to flag overshoot.

## Gain-prediction analysis (owner arm 2026-07-24): gain*(t) = α*(t) × ρ(t)

ρ(t) = per-timestep val R² (`TBIN_EVAL` in the trainers; `gain_predict.py`). Recorded 2026-07-24
11:20 UTC. **PROSPECTIVE v4 line recorded BEFORE reading any v4 sweep scores** (scores.json did not
exist yet — fresh probe confirmed):

| host / ckpt | α*(t) | ρ(t) | predicted gain*(t) | arm ranking by rms-distance |
|---|---|---|---|---|
| OD v2-v3 valpeak (retro) | 0.96 / 0.667 | 0.444 / 0.536 | 0.426 / 0.358 | corrgate050 (0.04) < corr050 (0.11) < corrgate025 (0.19) < corrgate < corr |
| OD v2-v4 valpeak (**prospective**) | 0.892 / 0.690 | 0.361 / 0.392 | 0.322 / 0.270 | **corrgate025 (0.098) ≈ corrgate050 (0.103)** < corr050 (0.21) < corrgate < corr — predicted optimum ≈ gate×0.36–0.39 |
| OD v2-v4 final (prospective) | 0.892 / 0.690 | 0.331 / 0.381 | 0.295 / 0.263 | same ordering, slightly closer to 025 |
| HY v2c2 (retro) | 0.81/0.53/0.53/0.58 | 0.51/0.54/0.54/0.47 | 0.42/0.29/0.29/0.27 | **corrgate050 (0.02)** < tsplit_a (0.11) < corrgate025 (0.16) < corrgate (0.29) < corr (0.63) |

Retrospective verdict: **MIXED** — HOLDS on HY (the prediction ranks the arms exactly as the owner
verdicts ordered them: 050 shipped-best, tsplit mixed, flat-025 fail, corrgate/corr worse); FAILS on
the OmniDreams eyeball ordering (prediction puts corrgate050 over corrgate025, the owner shipped 025) —
the predictor tracks drift-cut effectiveness while the OD eyeball optimum was artifact-free detail at
lower gain. Prospective v4 test OUTCOME (sweep landed after the prediction): top-1 HIT / near-tie MISS — overall verdict stays **MIXED**.

Verification notes (issue #1, 2026-07-23): scen6 = HF sample `239a869e-20a7-11ef-9e61-00044bf65d5c`
(never seen by training). Eval rollouts use NON-tiled real HDMaps, so the 40-frame recurrence cannot
come from the conditioning (HDMap autocorr shows no lag-40 peak). Whole-frame autocorrelation is NOT
diagnostic here (dominated by static road/sky layout) — use the roadside strips. The lag-40 comb
amplitude is small; the dominant measured signature is generic self-copying / novel-content
suppression, which is the same training shortcut (copy lap-aligned history) expressed off-period.
