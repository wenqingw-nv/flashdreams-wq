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
| 1 | **Progression under commanded motion** — full gain and plain α*(t) gate attenuate commanded trajectories (dynamics −30..46%; gate concentrates correction at high t where structure is decided → camera never reaches the bridge) | **corrgate050 PASSED its kill bar** (bridge cells): latesim 0.741 < corr050 0.779 (corrgate 0.873), MUSIQ 72.3/71.6 > corr050, Δ −0.15 (negative); soft spot dynamics −23% vs base | instruments pass on BOTH axes (bridge + static suite: dyn 0.92× base vs corrgate 0.78×); owner eyeball on sbs (`eval_sweep/sbs_*corrgate050*`, vs base and vs corr050) | — | verdict on sbs |
| 2 | **Corrector-induced seam pulse + post-boundary blur** — absent in the base (production-trained, boundary-clean); caused by piecewise-per-chunk correction jumps + ungated flat gain injecting the non-systematic low-t share (α* 0.81@t1000 → ~0.53 below) straight to pixels on the 4-step distilled solver | seam metrics re-aligned to the true 13+16k decoded cadence (old 13-alignment smeared the signal) + new seam_sharpness_ratio; **corrgate050**: pulse 1.243 (corr050 1.372, base 1.188), post-boundary sharpness 0.977 ≈ base 0.987 | much reduced on instruments (static suite: corrgate050 1.226/0.972 vs corr050 1.267/0.958, base 1.201/0.981); high-t-only fallback not needed so far; owner eyeball with #1 | — | with #1 |
| 3 | **v2 saturation overshoot** — sat-drift 0.070 vs base 0.045 (v1↔v2 DAgger oscillation) | **CLOSED (2026-07-21)**: v3 re-eval — no deploy point where v3 wins (full-gain sat 0.058 vs 0.066 but worse latesim/quality/pulse; v2 dominates at 0.5/gate); low-gain composition already resolves sat (0.013–0.019) | v2 stays the deployed LoRA; `eval_v3/scores.json` | — | closed |
| 4 | **Host-level jump-cut at chunk boundaries** — content-level context/memory handoff artifact, present in the base; NOT the corrector's pulse (#2) and not corrector-fixable | fix class: content-level overlap/blend in the runner | parked (host-level, out of corrector scope) | — | post-refresh |

Acceptance demo (not an issue): static-background suite **DONE** (8 scenes × {corr050, gate, gate×0.5}
+ base; 45 sbs + scores under `outputs/demo_static/`). Instruments: **corrgate050 = best all-round**
(dyn 0.92× base, seams ≈ base, Δ −0.13, +1.7 MUSIQ, sat halved); corrgate best quality (65.7) but
fails the motion guard (0.78×); corr050 keeps most motion (0.95×) with the strongest pulse/blur
(1.267/0.958). Entrance scene7: motion 0.91×/0.95× base for corrgate050/corr050 (corrgate 0.76×
suppresses). Owner eyeball decides ship config per use case.
