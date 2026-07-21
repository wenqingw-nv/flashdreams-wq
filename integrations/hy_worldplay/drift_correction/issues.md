<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay port — quality issues board

Deploy picture so far (owner-verified): **corr050 = commanded-motion default** (reaches the bridge,
better saturation than base) · **gate = static/drift-critical** (best quality MUSIQ 73.8, anchors).
Rules: gate first with pre-set kill bars · owner eyeball outranks instruments · failed arms live in
git history, not on this board.

| # | issue | latest fix | status | running now | ETA |
|---|---|---|---|---|---|
| 1 | **Progression under commanded motion** — full gain and plain α*(t) gate attenuate commanded trajectories (dynamics −30..46%; gate concentrates correction at high t where structure is decided → camera never reaches the bridge) | **corrgate050 PASSED its kill bar** (bridge cells): latesim 0.741 < corr050 0.779 (corrgate 0.873), MUSIQ 72.3/71.6 > corr050, Δ −0.15 (negative); soft spot dynamics −23% vs base | **OWNER EYEBALL PASS (2026-07-21, bridge sbs): corrgate050 CONFIRMED for commanded motion**, now backed by verified real-disk instruments (13:29): latesim 0.669, dyn 20.3 ≈ base (no dynamics loss — phantom "−23%" corrected), seam 1.211 ≤ base, Δ −48% vs base, MUSIQ 70.4; knee sweep NOT NEEDED | verified; ship decision on static-suite eyeball | owner static verdict |
| 2 | **Corrector-induced seam pulse + post-boundary blur** — absent in the base (production-trained, boundary-clean); caused by piecewise-per-chunk correction jumps + ungated flat gain injecting the non-systematic low-t share (α* 0.81@t1000 → ~0.53 below) straight to pixels on the 4-step distilled solver | seam metrics re-aligned to the true 13+16k decoded cadence (old 13-alignment smeared the signal) + new seam_sharpness_ratio; **corrgate050**: pulse 1.243 (corr050 1.372, base 1.188), post-boundary sharpness 0.977 ≈ base 0.987 | verified on real disk (bridge + static): corrgate050 seam 1.211/0.993 bridge, 1.362/0.980 static — mildest pulse of any corrector config, post-boundary sharpness ≈ base; high-t-only fallback not needed | verified | with #1 |
| 3 | **v2 saturation overshoot** — sat-drift 0.070 vs base 0.045 (v1↔v2 DAgger oscillation) | **CLOSED (2026-07-21, verified real-disk scores 13:23)**: v3's targeted sat overshoot got worse at full gain (0.074 vs v2 0.066); at gate×0.5 v3 trades small Δ/pulse gains for clearly worse dynamics (15.3 vs 20.3) and progression (0.741 vs 0.669) — v2 stays the deployed LoRA | `eval_v3/scores.json` | — | closed |
| 4 | **Host-level jump-cut at chunk boundaries** — content-level context/memory handoff artifact, present in the base; NOT the corrector's pulse (#2) and not corrector-fixable | fix class: content-level overlap/blend in the runner | parked (host-level, out of corrector scope) | — | post-refresh |

Acceptance demo (not an issue): static-background suite **scored on real disk (2026-07-21 13:17)** —
verified table in `TODO.md`. corrgate050 = best corrector config on instruments (dyn 0.87× base,
sharpness ≈ base, mildest pulse, entrance preserved at 0.86×); flags: corrector MUSIQ below base on
these scenes, scene5 motion collapse (all configs), corr050 erratic. Owner eyeball on
`demo_static/sbs_*` decides ship config per use case.
