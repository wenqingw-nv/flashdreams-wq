<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Upstream PR draft — hand back to owner (gh CLI not installed on this box)

**Base:** `NVIDIA/flashdreams` `main` · **Head:** `wenqingw-nv/flashdreams-wq:clean-forcing-hy-worldplay`
(single signed-off commit `58702aaf`, 20 files, +3355; rebased on `origin/main` @ `eadf2596`)

Create with:

```bash
gh pr create --repo NVIDIA/flashdreams \
  --base main --head wenqingw-nv:clean-forcing-hy-worldplay \
  --title "hy_worldplay: Clean Forcing drift corrector (training recipe + content-keyed deploy)" \
  --body-file integrations/hy_worldplay/drift_correction/PR_DRAFT.md   # body section below
```

## Title

hy_worldplay: Clean Forcing drift corrector (training recipe + content-keyed deploy)

## Body

Long autoregressive rollouts of the HY-WorldPlay WAN-5B integration drift:
each chunk conditions on self-generated history and small errors compound
into saturation/contrast runaway and texture mush. This PR adds a trained
drift corrector plus the full zero-real-data recipe that produced it.

### What's included

- **Deploy hook** (`hy_worldplay/_drift_corrector.py` + two runner-config
  fields): `drift_corrector` (LoRA checkpoint path) and
  `drift_corrector_gain` (default 0.5). Selection is content-keyed per job:
  static (locked-off camera) trajectories run the untouched base weights
  (static scenes measure *negative* drift on this host — correction there is
  pure artifact cost); commanded-motion trajectories apply the LoRA at
  `alpha*(t) x gain` per denoise step. `drift_corrector=None` (default) is
  byte-identical to the current behavior.
- **Training + eval recipe** (`drift_correction/`): strafe-loop pair
  construction, the step-0 `alpha*(t)` systematicity diagnostic (go/no-go
  gate and the deploy gain profile), v1 counterfactual-teacher training, v2
  DAgger + drift-contraction round, gain-sweep evaluation, drift/dynamics/
  progression/seam scoring, static-scene demo suite. README documents the
  method and the ~2.5 GPU-day reproduction path.
- **CPU unit tests** (`tests/test_drift_corrector.py`, `ci_cpu`): LoRA
  wrapper is a strict identity until a checkpoint is loaded and gated on;
  content-keyed selection triggers on action labels.

### Measured results (704x1280, 24-chunk / ~19 s rollouts, held-out trajectories)

| config | MUSIQ ↑ | Δ-drift (MUSIQ first-20% − all) | dynamic degree | 
|---|---|---|---|
| base | 68.0 | +1.70 | 19.3 |
| shipped `alpha*(t) x 0.5` | 70.4 | **−48%** | 20.3 (≈ base) |
| flat gain 1.0 | 72.1 | −133% (negative drift) | 11.7 (−40%) |

The shipped configuration cuts the drift metric ~50% with no dynamics loss
and no added chunk-boundary pulse; the base's late-horizon saturation
runaway is eliminated. Full-gain variants trade commanded-trajectory
progression for raw drift reduction and are not defaults.

### Notes for reviewers

- The corrector LoRA is 14.75M params (~0.3%); motion jobs keep it unfused
  because a single-scale weight merge cannot express the per-timestep
  `alpha*(t)` gate. Static jobs bypass module surgery entirely.
- The trained v2 checkpoint (~30 MB `.pt`) is distributed outside the repo;
  owner to attach as a release asset / HF artifact and link here.
- Scripts default to co-tenant-friendly settings (eager VAE, expandable
  segments) and are resumable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Pre-submit checklist (done on this box)

- [x] `ruff format` + `ruff check` (pinned v0.12.7) clean on all touched files
- [x] `pytest integrations/hy_worldplay/tests/test_drift_corrector.py` — 6 passed
- [x] DCO sign-off on the single squashed commit (`Signed-off-by: Wenqing Wang <wenqingw@nvidia.com>`)
- [x] SPDX headers on every new file
- [ ] `uv run pre-commit run -a` + `uv run pytest -m ci_cpu` full-repo pass (recommended before opening)
- [ ] Checkpoint distribution decision (release asset vs HF) — owner call
