<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Clean Forcing port — current work & next steps

*(2026-07-21 refresh. History → git log of this file; quality issues → `issues.md`; results →
`REPORT.md` + `outputs/*/scores.json`.)*

## Where we are

**Deploy candidate: `corrgate050` (α*(t) gate × 0.5 gain) — passed both kill bars on instruments.**
Bridge: best progression of any corrector config (latesim 0.741), seams < corr050, post-boundary
sharpness ≈ base, Δ-drift −0.15, MUSIQ 72.3. Static suite (8 scenes incl. entrance): motion guard
PASS (0.92× base; entrance scene 0.91×), sat halved (0.019), host jump-cuts 0.38→0.13, +1.7 MUSIQ.
Fallback split if owner rejects: corr050 for commanded motion, plain gate for static/drift-critical.
v3 re-eval CLOSED (no deploy point beats v2 — v2 stays the deployed LoRA). Seam metric fixed (true
cadence 13+16k; pre-fix seam numbers not comparable).

## Now (blocking on owner eyeball)

1. **Owner verdicts** on: bridge sbs (`outputs/eval_sweep/sbs_*corrgate050*`), static demo sbs
   (`outputs/demo_static/sbs_*scene{0..7}*`), and the scene-7 entrance. → ship decision.
2. **Owner call**: accept corrgate050's −23% dynamics on commanded motion, or sweep
   gate×{0.55–0.7} for the knee (~3 h GPU).

## Next (after ship decision)

3. Update `REPORT.md` deploy recommendation to the chosen config; merge the LoRA into weights
   behind a runner flag (zero-overhead deploy path).
4. Stop the stale parallel agent session (was launching duplicate jobs; three dups already killed).
5. Ops: add the ~45-min filesystem/clock desync between shells to the existing UVM driver ticket.

## Parked

- Host-level jump-cut at chunk boundaries (base artifact; content-level overlap/blend in the
  runner — not corrector-fixable). See `issues.md` row 4.
- Research arms (commanded-future teacher, pose-memory oracle): owner-gated, documented in
  `~/projs/drift_correction/flashdreams_value.md`.
- Official VBench 6-dim suite via the on-box harness (paper-comparable table) — nice-to-have.
