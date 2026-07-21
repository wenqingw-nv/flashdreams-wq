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

**Static-background suite — VERIFIED table (2026-07-21, scores.json 13:17 on disk; supersedes the
retracted phantom table, which was wrong in every number):**

| config | MUSIQ/late | Δ-drift | dyn (×base) | sat | cuts | latesim | seam-mot | seam-sharp |
|---|---|---|---|---|---|---|---|---|
| base | 53.2/56.3 | −5.97 | 12.8 (1.00) | 0.054 | 0.25 | 0.618 | 1.251 | 0.979 |
| corr050 | 47.9/45.6 | −0.64 | 19.6 (2.18, erratic) | 0.028 | 0 | 0.718 | 1.607 | 0.885 |
| corrgate | 51.4/52.4 | −3.10 | 6.2 (0.59) | 0.029 | 0 | 0.841 | 1.643 | 0.952 |
| **corrgate050** | 50.0/51.5 | −3.89 | 8.9 (0.87) | 0.024 | 0 | 0.818 | 1.362 | 0.980 |

Read: corrgate050 is the only corrector config near the motion guard (−13%) with sharpness ≈ base
and the mildest pulse; the stronger anchoring (latesim 0.82 vs base 0.62) is the desired behavior
on static scenes. Flags for the owner eyeball: (1) corrector configs sit below base MUSIQ here
(base Δ −5.97 — the base *improves* over these rollouts, unlike the bridge cells); (2) scene5
motion collapses under every corrector (0.06–0.16×); (3) corr050 is erratic on static scenes
(per-scene dyn 0.16–6.7×). Entrance scene7: corrgate050 dyn 0.86× base, seam 1.096 (best config),
+4.9 MUSIQ. Artifacts: `outputs/demo_static/{scores.json,sbs_*.mp4}`.

## Now

1. **Owner eyeball on the static-suite sbs** (`outputs/demo_static/sbs_*scene{0..7}*`), with the
   three flags above called out (esp. scene5 motion collapse + scene7 entrance).
2. v3 + eval_sweep real-view re-scores in flight; v2-vs-v3 verdict re-issues when v3 lands.
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
