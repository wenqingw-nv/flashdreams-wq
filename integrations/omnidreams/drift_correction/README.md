<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams drift corrector — SHIPPED config (2026-07-24)

Clean Forcing drift corrector for the OmniDreams single-view host.

## Ship decision (owner eyeball, 2026-07-24)

**Deploy `outputs/lora_v2_v3_valpeak.pt` at `alpha*(t) x 0.25` (the
`corrgate025` dial).** md5 `84beb8c014346833ce182310373b5d32`, 88,339,905
bytes (LoRA r16 on self-attn q/k/v/o, ~7.3M params; `train_v2.py` format).
Owner verdict: best of all arms — better trees/foliage detail and
consistency, less repeated street view; drift Delta +0.99 vs base +2.44
(60% cut), MUSIQ 50.1 > base 49.1, dynamics −17%.

## Quick start

```python
from omnidreams.runner import OmnidreamsRunnerConfig

cfg = OmnidreamsRunnerConfig(
    ...,
    drift_corrector=Path(
        "integrations/omnidreams/drift_correction/outputs/lora_v2_v3_valpeak.pt"
    ),
    # drift_corrector_gain defaults to the shipped 0.25
)
```

The runner hook (`omnidreams/_drift_corrector.py`) wraps the transformer's
self-attn projections with the LoRA and rescales it to
``alpha*(t) x gain`` before every denoise step (alpha* profile: 0.96 at
t=1000, 0.667 at t=803 — the pairs-v2 photoreal systematicity gate).

## Known residual issues (owner-accepted at ship)

- scen6 shows some repeated rows of buildings/shops and 3 repeated white
  trucks (partial repeat prior; fix attempts recorded in `issues.md` and
  git history).
- Slight softness vs base (sharpness_ratio 0.598 vs 0.869) — owner-accepted;
  note that sharpness_ratio is DISQUALIFIED as a kill-bar instrument on this
  host (mispredicted the owner verdict in both directions; use owner frame
  stacks for the sharpness axis).
- Dynamics −17% vs base.

## Layout

Training/eval scripts (`build_pairs*.py`, `train_v{1,2}.py`,
`gate_faithful.py`, `eval_rollouts.py`, `gain_predict.py`) + helpers
(`_host.py`, `_lora.py`, `_train_attn.py`). Available ablation flags:
`PAIR_SCHEME=fork` (pairs-v4 re-anchored fork counterparts; gate-validated
design, sweep failed owner eyeball) and `UW=1` (per-token alpha*-weighted
loss). History and decisions: `issues.md` + git log of this directory.
