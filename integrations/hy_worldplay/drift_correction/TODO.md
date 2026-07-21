<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Clean Forcing port — status

*(2026-07-21: SHIPPED. History → git log; issues → `issues.md`; results + ship decision →
`REPORT.md`; scores → `outputs/*/scores.json`.)*

## Shipped

Content-keyed deployment (owner decision 2026-07-21):
**commanded-motion jobs → corrgate050** (v2 LoRA at per-step α*(t)×0.5) ·
**static jobs → corrector OFF** (base weights; static scenes measure negative drift — correction
there is pure artifact cost, by design not a defect).

Runner flag: `HyWorldPlayWanI2VRunnerConfig.drift_corrector` (LoRA checkpoint path) +
`drift_corrector_gain` (default 0.5); selection logic in `hy_worldplay/_drift_corrector.py`
(job-level, keyed on the pose's action labels). Verified end-to-end on GPU: motion pose logs
"corrected (alpha*(t) x gain)", static pose logs "base (static trajectory)"
(`outputs/flag_verify/`). Note: motion jobs keep the LoRA unfused (~0.3% params of matmul) because
a single-scale weight merge cannot express the per-timestep gate; static jobs have zero overhead.

## Open (post-ship, low priority)

1. Stop the stale parallel agent session (was launching duplicate jobs; three dups killed).
2. Ops: add the shell filesystem/clock desync to the existing UVM driver ticket; all published
   numbers were re-verified on real disk after the phantom-view incident (see git log).
3. Official VBench 6-dim suite via the on-box harness (paper-comparable table) — nice-to-have.

## Parked

- Host-level jump-cut at chunk boundaries (base artifact; content-level overlap/blend in the
  runner — not corrector-fixable). `issues.md` row 4.
- Research arms (commanded-future teacher, pose-memory oracle): owner-gated, documented in
  `~/projs/drift_correction/flashdreams_value.md`.
