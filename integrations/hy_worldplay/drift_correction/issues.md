<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay port — quality issues board

**SHIPPED (owner decision 2026-07-21): content-keyed deployment — commanded-motion jobs =
corrgate050 (v2 LoRA at α*(t)×0.5 per step) · static jobs = corrector OFF (base weights; static
scenes measure negative drift — nothing to correct).** Runner flag:
`HyWorldPlayWanI2VRunnerConfig.drift_corrector` (job-level selection on trajectory content).
Rules: gate first with pre-set kill bars · owner eyeball outranks instruments · failed arms live in
git history, not on this board.

| # | issue | latest fix | status | running now | ETA |
|---|---|---|---|---|---|
| 1 | **Progression under commanded motion** — full gain and plain α*(t) gate attenuate commanded trajectories (dynamics −30..46%; gate concentrates correction at high t where structure is decided → camera never reaches the bridge) | **corrgate050 PASSED its kill bar** (bridge cells): latesim 0.741 < corr050 0.779 (corrgate 0.873), MUSIQ 72.3/71.6 > corr050, Δ −0.15 (negative); soft spot dynamics −23% vs base | **CLOSED — SHIPPED (2026-07-21)**: corrgate050 for commanded-motion jobs (owner eyeball PASS + verified instruments: latesim 0.669, dyn 20.3 ≈ base, seam 1.211 ≤ base, Δ −48%); runner flag `drift_corrector` keys it per job on trajectory content | `hy_worldplay/_drift_corrector.py` + `REPORT.md` ship decision | — | closed |
| 2 | **Corrector-induced seam pulse + post-boundary blur** — absent in the base (production-trained, boundary-clean); caused by piecewise-per-chunk correction jumps + ungated flat gain injecting the non-systematic low-t share (α* 0.81@t1000 → ~0.53 below) straight to pixels on the 4-step distilled solver | seam metrics re-aligned to the true 13+16k decoded cadence (old 13-alignment smeared the signal) + new seam_sharpness_ratio; **corrgate050**: pulse 1.243 (corr050 1.372, base 1.188), post-boundary sharpness 0.977 ≈ base 0.987 | **CLOSED — resolved by the ship rule (2026-07-21)**: gate×0.5 gives the mildest pulse of any corrector config (bridge 1.211/0.993 at/below base); residual static-scene pulses mooted by corrector-OFF-for-static | ship decision in `REPORT.md` | — | closed |
| 3 | **v2 saturation overshoot** — sat-drift 0.070 vs base 0.045 (v1↔v2 DAgger oscillation) | **CLOSED (2026-07-21, verified real-disk scores 13:23)**: v3's targeted sat overshoot got worse at full gain (0.074 vs v2 0.066); at gate×0.5 v3 trades small Δ/pulse gains for clearly worse dynamics (15.3 vs 20.3) and progression (0.741 vs 0.669) — v2 stays the deployed LoRA | `eval_v3/scores.json` | — | closed |
| 4 | **Host-level jump-cut at chunk boundaries** — content-level context/memory handoff artifact, present in the base; NOT the corrector's pulse (#2) and not corrector-fixable | fix class: content-level overlap/blend in the runner | parked (host-level, out of corrector scope) | — | post-refresh |
| 5 | **Person/stylized-content generalization** — OOD card-image cells (2026-07-22): people blur/morph vs base in person-walking content; stylized scenes regress on Delta/MUSIQ (corrector inherits its training content regime — test.png seeds contain no humans) | **v2c2 PASSED the kill bar (2026-07-22)**: continue-train from v2, 200 steps @1e-4, 8 card-image clips (3 person scenes) at 10% pool weight (first attempt v2c failed: uniform-over-pools = 25% weight diluted the bridge Delta-cut to ~0%). Result: bridge Delta -58% (restored), e_person Delta -47% / MUSIQ +5.4 / seams best, b_strafe -31%, a_turn holds | **CLOSED — SHIPPED (owner eyeball PASS 2026-07-22)**: v2c2 supersedes v2 for motion jobs | `outputs/eval_v2c2_*/` + `outputs/lora_v2c2.pt` (snapshots _step100/_step200) | eyeball | done |

| 6 | **corrgate025 dial-down eval (owner arm 2026-07-24)** — v2c2 at α*(t)×0.25 vs corrgate050 vs base on bridge (default 3 poses) + castle/flags + person card cells, seed 5042, 24 chunks. CAVEAT: the original castle/e_person ad-hoc cell poses/prompts were not persisted; those two cells were regenerated as matched within-run triples (seed image = frame 0 of the owner's cells; documented poses/prompts in the eval script) — absolute numbers not comparable to eval_v2c2 tables | **Pre-stated bars FAIL**: bridge Δ 1.53 vs base 1.18 (NO cut; 050 cuts 52% — retention bar ≥60% of 050's missed by a wide margin) · castle Δ 5.59 vs base 2.39 (worse than 050's 4.35) with visible saturation runaway in late frames (`eval_g025_castle/g025_recheck/castle_stack.png`; sharpness check inconclusive — reconstructed pose ends against a wall) · person Δ 6.70 between base 7.45 and 050's 4.39. Only win: dynamics preserved (bridge 24.6 ≈ base 20.9; castle 49.4 vs 050's 31.7) | **OPEN — dial-down NOT recommended by instruments** (α×0.25 loses the drift cut on this host, unlike OmniDreams where ×0.25 + val-peak ckpt shipped); owner eyeball on sbs decides; default gain stays 0.5 | `outputs/eval_g025_{bridge,castle,person}/` + sbs | — | owner eyeball |

Acceptance demo (not an issue): static-background suite **scored + owner-eyeballed (2026-07-21)** —
verdict: **base wins both axes on static content** (quality and pulse ordering
base > corrgate050 > corrgate > corr full). Consistent with the instruments (base Δ −5.97 —
static scenes don't drift on this host). **Ship rule: corrector OFF for static jobs — by design,
not a failure to fix**; scene-5 motion collapse is moot under this rule. Verified table in git
history of `TODO.md`; sbs under `demo_static/`.
