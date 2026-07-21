<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Clean Forcing port — current work & next steps

*(2026-07-21 refresh. History → git log of this file; quality issues → `issues.md`; results →
`REPORT.md` + `outputs/*/scores.json`.)*

## Where we are

**Deploy candidate: `corrgate050` (α*(t) gate × 0.5 gain) — passed the BRIDGE kill bar per
`eval_sweep/scores.json`, now UNDER RE-VERIFICATION:** the file is on disk but was computed in the
desynced view (deterministic seam metrics differ between it and a real-view pass on identical base
videos: 1.188 vs 1.249), so a real-view re-score of eval_sweep is queued; bridge conclusions are
provisional until it lands. Seam metric fix (true cadence 13+16k) is code-level and stands.

**RETRACTION (2026-07-21):** the static-suite table and the v3-re-eval closure previously recorded
here were read through a desynced agent filesystem view; neither scores.json existed on disk.
Real state: static suite corrgate/corrgate050 cells GENERATING now (base + corr050 done); v3
rollouts + sbs are on disk but unscored. Verified numbers replace this note when scoring lands.

## Now

1. Static suite: wait out the live generation (do not restart), then score
   {base, corr050, corrgate, corrgate050}, build sbs for all 8 scenes, re-issue the table verified.
2. Score `eval_v3` (rollouts on disk) and re-issue the v2-vs-v3 verdict verified.
3. **Bridge OWNER EYEBALL: PASS (2026-07-21)** — corrgate050 "much better in progression, without
   much pulling back or pulse issue" (both sbs comparisons). Confirmed for commanded motion; −23%
   dynamics accepted; gate×{0.55–0.7} knee sweep NOT NEEDED unless the static suite objects.
4. Ship decision waits only on the static suite (owner eyeball on its sbs once real). If static
   confirms: corrgate050 = single ship config → REPORT.md deploy recommendation + LoRA merge
   behind a runner flag.

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
