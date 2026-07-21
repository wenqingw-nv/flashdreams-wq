<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Clean Forcing port — current work & next steps

*(2026-07-21 refresh. History → git log of this file; quality issues → `issues.md`; results →
`REPORT.md` + `outputs/*/scores.json`.)*

## Where we are

**Deploy candidate: `corrgate050` (α*(t) gate × 0.5 gain) — bridge kill bar PASSED on verified
real-disk scores (13:29) + OWNER EYEBALL PASS.** Verified bridge row: latesim 0.669 (best of any
corrector config; corr050 0.779, corrgate 0.873), dyn 20.3 ≈ base 19.3 (**no dynamics loss** — the
earlier "−23%" was a phantom-view artifact), seam 1.211 / sharp 0.993 (mildest pulse, at/below
base), MUSIQ 70.4 (> base 68.0, marginally < corr050 71.2), Δ +0.88 = −48% cut vs base (bar ≥30%).
corrgate stays the pure-quality point (73.8, Δ −151%) at heavy anchoring. Seam metric fix (true
cadence 13+16k) stands.

**v3 re-eval, verified (13:23): v2 stays deployed.** v3's targeted saturation overshoot got worse
at full gain (0.074 vs v2 0.066); at gate×0.5 v3 trades slightly better Δ/pulse (+0.38/1.175) for
clearly worse dynamics (15.3 vs 20.3) and progression (latesim 0.741 vs 0.669) — the owner's
priority axes. `eval_v3/scores.json`.

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
