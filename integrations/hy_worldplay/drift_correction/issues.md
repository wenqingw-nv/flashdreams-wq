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
| 1 | **Progression under commanded motion** — full gain and plain α*(t) gate attenuate commanded trajectories (dynamics −30..46%; gate concentrates correction at high t where structure is decided → camera never reaches the bridge) | **corrgate050 PASSED its kill bar** (bridge cells): latesim 0.741 < corr050 0.779 (corrgate 0.873), MUSIQ 72.3/71.6 > corr050, Δ −0.15 (negative); soft spot dynamics −23% vs base | **OWNER EYEBALL PASS (2026-07-21, bridge sbs, both comparisons): "much better in progression, without much pulling back or pulse issue" — corrgate050 CONFIRMED for commanded motion**; −23% dynamics accepted, gate×{0.55–0.7} knee sweep NOT NEEDED unless static suite objects; instrument re-score of eval_sweep still queued (view-desync hygiene) | static-suite rollouts + re-scores (demo, sweep, v3) | ship decision on static suite |
| 2 | **Corrector-induced seam pulse + post-boundary blur** — absent in the base (production-trained, boundary-clean); caused by piecewise-per-chunk correction jumps + ungated flat gain injecting the non-systematic low-t share (α* 0.81@t1000 → ~0.53 below) straight to pixels on the 4-step distilled solver | seam metrics re-aligned to the true 13+16k decoded cadence (old 13-alignment smeared the signal) + new seam_sharpness_ratio; **corrgate050**: pulse 1.243 (corr050 1.372, base 1.188), post-boundary sharpness 0.977 ≈ base 0.987 | much reduced on the bridge cells (verified); static-suite seam numbers RETRACTED (phantom view), re-score pending; high-t-only fallback not needed so far | (same regeneration as #1) | with #1 |
| 3 | **v2 saturation overshoot** — sat-drift 0.070 vs base 0.045 (v1↔v2 DAgger oscillation) | v3 re-eval rollouts + sbs on disk (`eval_v3/`); the earlier "closed, v2 wins" numbers are RETRACTED (read through a phantom view — no scores.json existed); scoring queued | reopened pending verified score | eval_v3 scoring | with static-suite scoring |
| 4 | **Host-level jump-cut at chunk boundaries** — content-level context/memory handoff artifact, present in the base; NOT the corrector's pulse (#2) and not corrector-fixable | fix class: content-level overlap/blend in the runner | parked (host-level, out of corrector scope) | — | post-refresh |

Acceptance demo (not an issue): static-background suite — previously posted "DONE" table RETRACTED
(phantom-view read; no scores.json on disk). Real state: base + corr050 cells on disk, corrgate +
corrgate050 generating now; scoring + sbs + verified table follow. Owner eyeball decides ship config
per use case.
