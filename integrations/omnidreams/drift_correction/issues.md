<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams port — quality issues board

**OWNER VERDICT (2026-07-23, eyeball on the sweep sbs): corrgate050 with v2@1000
(`outputs/lora_v2.pt` final, `eval_sweep_v2f`) is best overall — better than v2@600
(`eval_sweep_v2e`) and v1 (`eval_sweep`). This OVERRULES the instruments, which prefer v2@600
(corrgate050 Δ −1.34 vs v2@1000's −0.45; owner eyeball outranks instruments per house rules).**
Deploy recommendation: **PROVISIONAL — v2@1000 merged at α*(t)×0.5 (corrgate050)**; owner chose
RETRAIN (pairs-v3, issue #1 fix) before ship — recommendation stands only if the retrain does not
supersede it. Rules as on the HY board: gate first with pre-set kill bars · owner eyeball outranks
instruments · failed arms live in git history, not on this board.

| # | issue | latest fix | status | running now | ETA |
|---|---|---|---|---|---|
| 1 | **Repeat-prior artifact (owner class b)** — in `eval_sweep_v2f/...corrgate050_RIGHT_scen6_s5042.mp4`: missing zebra-crossing lines, repeated identical trucks/trailers lining both roadsides, repeated buildings. **Verified 2026-07-23, evidence FOR the repeat-prior mechanism** (training pairs tiled a FIXED 40-frame HDMap lap ×16, `build_pairs.py` — same failure class as HY's v2 repeat prior, fixed there by mixed LEGS lap lengths): (i) scen6 HDMap carries explicit crosswalk polygons + only sparse vehicle boxes at f280–640; base renders the zebra crossing and varied roadside buildings, corrgate050 suppresses the crossing and hallucinates a continuous band of identical trailers on both sides; (ii) roadside-strip temporal autocorrelation (late window, f280+): corrected self-similarity elevated at ALL lags (0.47–0.51 vs base 0.31–0.39 — novel-content suppression), with detrended bumps specifically at the 40/80/120-frame lap harmonics (mean +0.0037 at 40-multiples vs −0.0002 at non-multiples; NOT at generic 8-frame chunk multiples −0.0003; base flat ≈0). Evidence frames: `outputs/eval_sweep_v2f/repeat_prior_evidence/` | pairs-v3: mixed tile lengths (32/40/48-frame laps rotated per clip) break the fixed revisit period so "content recurs at lag 40" stops being a learnable shortcut; real photo seeds + lap-2 clean reference carried over from pairs-v2 | **OPEN — retrain GO (owner 2026-07-23)**: pairs-v3 → v1-v3 → DAgger+v2-v3 → sweep (must include scen6 + collapse scene; explicit repeat/crosswalk re-check) | pairs-v3 pipeline | ~7–9 h |
| 2 | **Fine-organic-detail blur + semantic slide (owner class a)** — occasional object blur and semantic drift on fine organic detail (blurry trees; leaves going full→bare) vs base | no dedicated fix designed; expected to improve with pairs-v3 (less copy-prior pressure) — re-check on the v3 sweep before designing anything | OPEN — re-assess after issue #1 retrain | — | with #1 |

Verification notes (issue #1, 2026-07-23): scen6 = HF sample `239a869e-20a7-11ef-9e61-00044bf65d5c`
(never seen by training). Eval rollouts use NON-tiled real HDMaps, so the 40-frame recurrence cannot
come from the conditioning (HDMap autocorr shows no lag-40 peak). Whole-frame autocorrelation is NOT
diagnostic here (dominated by static road/sky layout) — use the roadside strips. The lag-40 comb
amplitude is small; the dominant measured signature is generic self-copying / novel-content
suppression, which is the same training shortcut (copy lap-aligned history) expressed off-period.
