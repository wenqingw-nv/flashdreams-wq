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
| 1 | **Progression under commanded motion** — full gain and plain α*(t) gate attenuate commanded trajectories (dynamics −30..46%; gate concentrates correction at high t where structure is decided → camera never reaches the bridge) | **corrgate050** (α*(t) gate × 0.5 global gain) — kill bar: reach the bridge (sim-to-start ≈ corr050's 0.56 or better, owner eyeball final) with seam no worse than corr050; if it anchors, corr050 stays the pick and per-step t-restriction is the next dial | corr050 already meets the acceptance bar; corrgate050 tests whether one config can win both axes | ✅ corrgate050 bridge rollouts (3) | sbs + verdict ~hours |
| 2 | **Corrector-induced seam pulse + post-boundary blur** — absent in the base (production-trained, boundary-clean); caused by piecewise-per-chunk correction jumps + ungated flat gain injecting the non-systematic low-t share (α* 0.81@t1000 → ~0.53 below) straight to pixels on the 4-step distilled solver | same **corrgate050** run (gate kills the low-t noise, ×0.5 halves the per-chunk jump); fallback: correct only the first 1–2 high-t solver steps per chunk | seam_motion_ratio metric in scorer; 18 sbs under eval_sweep/ | ✅ (same run as #1) | with #1 |
| 3 | **v2 saturation overshoot** — sat-drift 0.070 vs base 0.045 (v1↔v2 DAgger oscillation) | **v3 (round-2 DAgger)** re-eval vs v2 | trained (`lora_v3.pt`); eval in flight | ✅ v3 re-eval (was 1/9) | ~hours |
| 4 | **Host-level jump-cut at chunk boundaries** — content-level context/memory handoff artifact, present in the base; NOT the corrector's pulse (#2) and not corrector-fixable | fix class: content-level overlap/blend in the runner | parked (host-level, out of corrector scope) | — | post-refresh |

Acceptance demo (not an issue): static-background suite ≥6 scenes at {corr050, gate, gate×0.5} + base,
incl. one new-element entrance scene — queued behind #1/#3; owner eyeball decides ship config per use case.
