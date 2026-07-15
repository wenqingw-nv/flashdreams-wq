# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drift + guard scoring for paired eval rollouts (Delta-MUSIQ, dynamics, strips).

Per MP4: per-frame MUSIQ (pyiqa) at a fixed frame stride, Delta-drift =
MUSIQ(first 20%) - MUSIQ(all) (BAgger's metric), a RAFT mean-flow dynamic
degree (guard: a corrector that freezes motion "wins" every consistency
metric), and a 10-frame contact strip PNG. Aggregates base-vs-corr into
``outputs/eval/scores.json``.

pyiqa is not part of the project deps; run with an ephemeral overlay::

    uv run --with pyiqa python integrations/hy_worldplay/drift_correction/score_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import os

EVAL_DIR = Path(
    os.environ.get(
        "EVAL_OUT", "integrations/hy_worldplay/drift_correction/outputs/eval"
    )
)
FRAME_STRIDE = 13
"""Score every 13th frame (one per AR chunk at the decoded rate)."""


def read_frames(path: Path, stride: int) -> np.ndarray:
    """Decode every ``stride``-th frame of an MP4 to ``[N, H, W, 3]`` uint8."""
    import imageio.v3 as iio

    frames = iio.imread(path, plugin="pyav")
    return frames[::stride]


def musiq_curve(frames: np.ndarray, model, device) -> list[float]:
    """Per-frame MUSIQ scores."""
    scores = []
    for f in frames:
        t = torch.from_numpy(f.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        scores.append(float(model(t)))
    return scores


def sharpness_curve(frames: np.ndarray) -> list[float]:
    """Per-frame variance-of-Laplacian (grayscale) — catches blur/structure
    loss that MUSIQ under-penalizes (observed on v1-pilot rollouts)."""
    from scipy.ndimage import laplace

    out = []
    for f in frames:
        gray = f.astype(np.float32).mean(axis=-1)
        out.append(float(laplace(gray).var()))
    return out


def advance_score(frames: np.ndarray) -> float:
    """1 - mean late-half frame-correlation to frame 0.

    On a forward trajectory the view should decorrelate from the start;
    a corrector with a learned repeat prior plateaus instead (observed:
    base 0.58 vs loop-trained corrector 0.38 on the bridge cells)."""
    g = frames.astype(np.float32).mean(axis=-1)[:, ::4, ::4]
    f0 = g[0] - g[0].mean()
    sims = []
    for f in g[len(g) // 2 :]:
        fc = f - f.mean()
        sims.append(
            float((f0 * fc).sum() / (np.linalg.norm(f0) * np.linalg.norm(fc) + 1e-8))
        )
    return float(1 - np.mean(sims))


def sat_drift(frames: np.ndarray) -> float:
    """Mean |saturation - saturation(frame 0)| (the reference's in-domain
    drift metric); catches the color-drift axis MUSIQ tolerates."""
    f = frames.astype(np.float32) / 255
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0).mean(axis=(1, 2))
    return float(np.abs(sat - sat[0]).mean())


def dynamic_degree(frames: np.ndarray, raft, device) -> float:
    """Mean RAFT flow magnitude between scored frames (motion guard)."""
    import torch.nn.functional as F

    mags = []
    for a, b in zip(frames[:-1], frames[1:]):
        ta = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().to(device)
        tb = torch.from_numpy(b.copy()).permute(2, 0, 1)[None].float().to(device)
        ta = F.interpolate(ta, size=(352, 640), mode="bilinear") / 127.5 - 1
        tb = F.interpolate(tb, size=(352, 640), mode="bilinear") / 127.5 - 1
        flow = raft(ta, tb)[-1]
        mags.append(float(flow.square().sum(1).sqrt().mean()))
    return float(np.mean(mags))


def contact_strip(frames: np.ndarray, out_path: Path, n: int = 10) -> None:
    """Write an n-frame horizontal strip PNG for eyeballing."""
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    strip = np.concatenate([frames[i] for i in idx], axis=1)
    Image.fromarray(strip).save(out_path)


def main() -> None:
    import pyiqa
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    device = "cuda" if torch.cuda.is_available() else "cpu"
    musiq = pyiqa.create_metric("musiq", device=device)
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    results: dict[str, dict] = {}
    configs = [d.name for d in sorted(EVAL_DIR.iterdir()) if d.is_dir()]
    for config in configs:
        cfg_dir = EVAL_DIR / config
        per_video = {}
        for mp4 in sorted(cfg_dir.glob("*.mp4")):
            if mp4.stem.startswith("sbs_"):
                continue
            frames = read_frames(mp4, FRAME_STRIDE)
            curve = musiq_curve(frames, musiq, device)
            sharp = sharpness_curve(frames)
            n20 = max(1, len(curve) // 5)
            delta = float(np.mean(curve[:n20]) - np.mean(curve))
            # Late-vs-early sharpness ratio: ~1.0 = detail preserved; well
            # below 1 = the compounding-blur failure mode.
            sharp_ratio = float(np.mean(sharp[-n20:]) / (np.mean(sharp[:n20]) + 1e-9))
            with torch.no_grad():
                dyn = dynamic_degree(frames, raft, device)
            contact_strip(frames, mp4.with_suffix(".strip.png"))
            per_video[mp4.stem] = {
                "musiq_overall": float(np.mean(curve)),
                "musiq_late": float(np.mean(curve[-n20:])),
                "delta_drift": delta,
                "dynamic_degree": dyn,
                "sharpness_ratio": sharp_ratio,
                "sat_drift": sat_drift(frames),
                "advance_score": advance_score(frames),
                "curve": curve,
                "sharpness": sharp,
            }
            print(
                f"{config}/{mp4.stem}: MUSIQ {np.mean(curve):.1f} "
                f"(late {np.mean(curve[-n20:]):.1f}) | Delta {delta:+.2f} | "
                f"dyn {dyn:.1f} | sharp-ratio {sharp_ratio:.2f}",
                flush=True,
            )
        agg = {
            key: float(np.mean([v[key] for v in per_video.values()]))
            for key in (
                "musiq_overall",
                "musiq_late",
                "delta_drift",
                "dynamic_degree",
                "sharpness_ratio",
                "sat_drift",
                "advance_score",
            )
        }
        results[config] = {"videos": per_video, "aggregate": agg}

    (EVAL_DIR / "scores.json").write_text(json.dumps(results, indent=2))
    print("\n================ closed-loop drift eval ================")
    print(
        f"{'':10s} {'MUSIQ':>7s} {'late':>7s} {'Delta':>7s} {'dyn':>6s} "
        f"{'sharp':>6s} {'sat':>7s} {'adv':>5s}"
    )
    for name in configs:
        a = results[name]["aggregate"]
        print(
            f"{name:10s} {a['musiq_overall']:7.2f} {a['musiq_late']:7.2f} "
            f"{a['delta_drift']:+7.2f} {a['dynamic_degree']:6.1f} "
            f"{a['sharpness_ratio']:6.2f} {a['sat_drift']:7.4f} "
            f"{a['advance_score']:5.2f}"
        )
    b = results.get("base", {}).get("aggregate")
    for name in configs:
        if name == "base" or b is None or b["delta_drift"] <= 0:
            continue
        red = 100 * (b["delta_drift"] - results[name]["aggregate"]["delta_drift"]) / b["delta_drift"]
        print(f"{name}: Delta-drift reduction vs base: {red:.0f}% (target >= 30%)")
    print(
        "guards: dynamic degree within ~20% of base (motion freeze) and "
        "sharpness ratio not far below base (compounding blur)."
    )


if __name__ == "__main__":
    main()
