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

Verification notes (issue #1, 2026-07-23): scen6 = HF sample `239a869e-20a7-11ef-9e61-00044bf65d5c`
(never seen by training). Eval rollouts use NON-tiled real HDMaps, so the 40-frame recurrence cannot
come from the conditioning (HDMap autocorr shows no lag-40 peak). Whole-frame autocorrelation is NOT
diagnostic here (dominated by static road/sky layout) — use the roadside strips. The lag-40 comb
amplitude is small; the dominant measured signature is generic self-copying / novel-content
suppression, which is the same training shortcut (copy lap-aligned history) expressed off-period.
