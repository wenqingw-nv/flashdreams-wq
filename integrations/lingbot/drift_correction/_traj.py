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

"""Pose-grammar trajectory generator for LingBot drift pairs.

LingBot has no pose-string grammar (poses come as ``.npy`` files), so this
module provides one for the drift-correction experiments, integrating
key-style commands into ``[T, 4, 4]`` camera-to-world poses with the same
:class:`~flashdreams.serving.webrtc.controls.CameraPoseIntegrator` the
interactive demo uses (RDF, 0.8 m/s move, 32 deg/s rotate at 16 fps).

Grammar: comma-separated ``<keys>-<frames>`` tokens, e.g.
``"w-24, wa-12, w-24, stay-6"`` — each token holds the key set for that many
frames. Keys are the demo's ``w/s`` (forward/back), ``a/d`` (yaw), ``q/e``
(strafe), ``i/k`` (pitch); ``stay`` (or ``x``) is a no-op hold.
"""

from __future__ import annotations

import numpy as np

from flashdreams.serving.webrtc.controls import CameraPoseIntegrator

FPS = 16
"""Pixel-frame rate the pose timeline is integrated at (LingBot default)."""

_VALID_KEYS = frozenset("wsadqeikjl")


def parse_grammar(grammar: str) -> list[frozenset[str]]:
    """Expand a pose-grammar string into one key set per pixel frame.

    Args:
        grammar: Comma-separated ``<keys>-<frames>`` tokens; ``stay``/``x``
            hold the camera still.

    Returns:
        One ``frozenset`` of active keys per frame.

    Raises:
        ValueError: A token is malformed or uses an unsupported key.
    """
    states: list[frozenset[str]] = []
    for raw in grammar.split(","):
        token = raw.strip()
        if not token:
            continue
        keys_part, sep, count_part = token.rpartition("-")
        if not sep or not count_part.isdigit():
            raise ValueError(f"malformed token {token!r} (want '<keys>-<frames>')")
        n = int(count_part)
        if keys_part in ("stay", "x"):
            keys: frozenset[str] = frozenset()
        else:
            keys = frozenset(keys_part)
            if not keys or not keys <= _VALID_KEYS:
                raise ValueError(f"unsupported keys in token {token!r}")
        states.extend([keys] * n)
    if not states:
        raise ValueError(f"empty grammar {grammar!r}")
    return states


def integrate_poses(
    states: list[frozenset[str]],
    initial_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate per-frame key states into ``[T, 4, 4]`` c2w poses.

    Frame ``t`` is the pose *after* applying state ``t`` for one frame
    interval; the trajectory starts one step past ``initial_pose`` (matching
    the interactive demo, whose resampler emits frame times strictly after
    the chunk start).

    Args:
        states: Per-frame key sets from :func:`parse_grammar`.
        initial_pose: Starting c2w pose; ``None`` = identity.
    """
    integ = CameraPoseIntegrator()
    integ.reset(initial_pose)
    dt = 1.0 / FPS
    poses = np.empty((len(states), 4, 4), dtype=np.float32)
    for t, keys in enumerate(states):
        t0 = t * dt
        poses[t] = integ.integrate_chunk(
            segments=[(t0, t0 + dt, keys)],
            frame_times=[t0 + dt],
        )[0]
    return poses


def grammar_poses(grammar: str, initial_pose: np.ndarray | None = None) -> np.ndarray:
    """Parse ``grammar`` and integrate it into ``[T, 4, 4]`` c2w poses."""
    return integrate_poses(parse_grammar(grammar), initial_pose)


## Pair-design trajectory families

def strafe_loop_grammar(legs: tuple[int, ...], wiggle: bool = False) -> str:
    """Strafe out-and-back loop with mixed leg geometry.

    Each lap strafes right ``leg`` frames then left ``leg`` frames — the
    camera returns to its starting pose with the *same orientation*, so
    lap-1 views are exact clean counterparts for later laps (the HY pair
    design). Mixed ``legs`` lengths are deliberate: uniform loops taught
    HY's corrector a repeat prior (owner-confirmed failure mode) — vary the
    geometry between clips, never ship a single fixed loop.

    Args:
        legs: Strafe leg length in frames for each lap.
        wiggle: Add a small alternating yaw during the legs (``ea``/``qd``)
            for extra view-geometry diversity; the yaw cancels within the
            lap so revisits still align approximately.
    """
    tokens = []
    for i, leg in enumerate(legs):
        out, back = ("e", "q") if i % 2 == 0 else ("q", "e")
        if wiggle:
            half = leg // 2
            tokens += [
                f"{out}a-{half}",
                f"{out}d-{leg - half}",
                f"{back}d-{half}",
                f"{back}a-{leg - half}",
            ]
        else:
            tokens += [f"{out}-{leg}", f"{back}-{leg}"]
    return ", ".join(tokens)


def repeat_intrinsics(row: np.ndarray, n_frames: int) -> np.ndarray:
    """Tile a single ``[4]`` fx/fy/cx/cy row into the ``[T, 4]`` layout."""
    row = np.asarray(row, dtype=np.float32).reshape(4)
    return np.tile(row[None, :], (n_frames, 1))
