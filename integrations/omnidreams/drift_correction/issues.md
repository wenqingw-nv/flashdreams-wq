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
1. **RUNNING — inference-time low-pass correction** (no retrain): at solver steps,
   `v_rect = v_base + α*(t)·gain·GaussianBlur_σ(v_corr − v_base)` (blur the correction delta only,
   latent-spatial, σ ∈ {1, 2}; `LP_SIGMA` in `eval_rollouts.py`). Corrector = v2-v3 val-peak at
   corrgate050 (final ckpt if budget allows), scenes {0, 6, 7} + healthy scen5 vs base. Kill bars
   (pre-stated): sharpness ≥ 0.757 (v2's level; ideally ≈ base 0.869) · Δ-drift cut within ~10% of
   un-low-passed v3 · explicit: trees stop disappearing (scen6)? leaf pulse gone (scen7)? Low-pass
   CANNOT fix the repeated-street-view artifact — recorded but not counted against these bars.
2. **HOLD for owner GO — pairs-v4, non-looping conditioning**: stitch distinct HDMap segments per
   training rollout (no lap tiling; content never recurs). Available: 32 single-view HF clips x ~80 s
   authentic HDMap (~2400 frames) — a single clip already covers a full 645-frame rollout untiled, and
   segments can be drawn across clips without replacement. Open design point: without revisits there is
   no lap-aligned clean counterpart — candidate replacement is segment-fresh short rollouts (low drift by
   construction) replayed at the long rollout's absolute indices via the existing forged-index machinery;
   seed inheritance at segment boundaries is the known risk to gate first. Estimated cost: pairs ~1.5-2 h
   (2x rollouts) · v1+v2 retrain ~5-6 h · sweep+scoring ~1.5 h => ~9-10 h end-to-end.

Checkpoints: `outputs/lora_v1_v3.pt`, `outputs/lora_v2_v3{,_valpeak,_stepN}.pt`; sbs (sides in
filename) under `outputs/eval_sweep_v3{,p}/`.

Verification notes (issue #1, 2026-07-23): scen6 = HF sample `239a869e-20a7-11ef-9e61-00044bf65d5c`
(never seen by training). Eval rollouts use NON-tiled real HDMaps, so the 40-frame recurrence cannot
come from the conditioning (HDMap autocorr shows no lag-40 peak). Whole-frame autocorrelation is NOT
diagnostic here (dominated by static road/sky layout) — use the roadside strips. The lag-40 comb
amplitude is small; the dominant measured signature is generic self-copying / novel-content
suppression, which is the same training shortcut (copy lap-aligned history) expressed off-period.
