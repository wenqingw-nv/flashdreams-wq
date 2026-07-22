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

"""Official VBench 6-dim scoring (calibrated scale) for the port's eval videos.

Scores the EXISTING rollouts — no generation. Uses the paper setup's
custom-input harness (on-box VBench checkout + Self-Forcing's decord shim);
must run from a neutral cwd in the ``df-gb300`` conda env::

    /localhome/local-wenqingw/miniconda3/envs/df-gb300/bin/python \
        integrations/hy_worldplay/drift_correction/score_vbench.py

Writes per-dimension raw JSONs and ``table.json`` under
``outputs/vbench/``; groups: ``motion_*`` = bridge cells (eval_sweep),
``static_*`` = the 8 locked-off demo scenes (demo_static).
"""

import glob
import json
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

SF = "/localhome/local-wenqingw/projs/Self-Forcing"
VB = "/localhome/local-wenqingw/projs/drift_correction/benchmarks/vbenchs/VBench"
sys.path.insert(0, VB)
sys.path.insert(0, SF)
import vbench_decord_shim  # noqa: E402

vbench_decord_shim.install()
import torch  # noqa: E402
from vbench import VBench  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs", "vbench")

DIMS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
]

CAL = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
}
"""Paper-table calibration bounds (VBench's official min/max per dim)."""

CONFIGS = ["base", "corr050", "corrgate", "corrgate050"]

GROUPS = {
    "motion": os.path.join(BASE, "outputs", "eval_sweep"),
    "static": os.path.join(BASE, "outputs", "demo_static"),
}


def link_videos(tag: str, src_glob: str) -> str:
    """Symlink the tag's videos into a per-tag dir and return the dir."""
    vdir = os.path.join(OUT, f"videos_{tag}")
    os.makedirs(vdir, exist_ok=True)
    for link in glob.glob(f"{vdir}/*.mp4"):
        if not os.path.exists(os.path.realpath(link)):
            os.remove(link)
    vids = sorted(glob.glob(src_glob))
    assert vids, f"no videos for {tag} at {src_glob}"
    for v in vids:
        dst = os.path.join(vdir, os.path.basename(v))
        if not os.path.lexists(dst):
            os.symlink(v, dst)
    return vdir


def main() -> None:
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    table_path = os.path.join(OUT, "table.json")
    tab = json.load(open(table_path)) if os.path.exists(table_path) else {}

    vb = VBench(
        torch.device("cuda"),
        os.path.join(VB, "vbench", "VBench_full_info.json"),
        os.path.join(OUT, "results"),
    )
    for group, src_dir in GROUPS.items():
        for config in CONFIGS:
            tag = f"{group}_{config}"
            vdir = link_videos(tag, os.path.join(src_dir, config, "*.mp4"))
            tab.setdefault(tag, {})
            for dim in DIMS:
                if dim in tab[tag]:
                    print(f"SKIP {tag} {dim} (cached)", flush=True)
                    continue
                try:
                    vb.evaluate(
                        videos_path=vdir,
                        name=f"{tag}_{dim}",
                        dimension_list=[dim],
                        mode="custom_input",
                    )
                    r = json.load(
                        open(os.path.join(OUT, "results", f"{tag}_{dim}_eval_results.json"))
                    )
                    raw = r[dim][0]
                    lo, hi = CAL[dim]
                    tab[tag][dim] = {
                        "raw": raw,
                        "calibrated": 100 * (raw - lo) / (hi - lo),
                    }
                    print(
                        f"RESULT {tag} {dim}: {tab[tag][dim]['calibrated']:.2f}",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001 — score remaining dims, report the failure
                    print(f"FAIL {tag} {dim}: {str(e)[:160]}", flush=True)
                json.dump(tab, open(table_path, "w"), indent=1)
    print("VBENCH-PORT-DONE", flush=True)


if __name__ == "__main__":
    main()
