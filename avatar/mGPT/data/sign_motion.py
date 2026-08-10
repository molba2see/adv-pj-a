"""Utilities for the disaster-safety Korean sign-language keypoint data.

The MotionGPT HumanML3D pipeline uses SMPL features.  This module keeps the
sign-language representation independent: body pose + both hands (201 values)
or, optionally, the face landmarks as well.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np


POSE = "pose_keypoints_3d"
LEFT_HAND = "hand_left_keypoints_3d"
RIGHT_HAND = "hand_right_keypoints_3d"
FACE = "face_keypoints_3d"


def find_dataset_root(root: str | Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    # Do not depend on Korean directory names: they can be renamed or appear
    # mojibake on Linux mounts.  Find a JSON whose schema identifies this data.
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "landmarks" in data and "sign_script" in data:
            return path.parent.parent
    raise FileNotFoundError(
        f"Could not find sign JSON files containing landmarks/sign_script below {root}. "
        "Check --data-root and confirm the JSON files were copied.")


def json_dir(root: str | Path) -> Path:
    root = find_dataset_root(root)
    candidates = []
    for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        valid = 0
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "landmarks" in data and "sign_script" in data:
                valid += 1
        if valid:
            candidates.append((valid, directory))
    if not candidates:
        raise FileNotFoundError(f"Could not find sign JSON files below {root}")
    return max(candidates, key=lambda item: item[0])[1]


def feature_keys(include_face: bool = False) -> list[str]:
    return [POSE, LEFT_HAND, RIGHT_HAND] + ([FACE] if include_face else [])


def load_keypoints(path: str | Path, include_face: bool = False) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    landmarks = data.get("landmarks", {})
    arrays = []
    frame_count = None
    for key in feature_keys(include_face):
        values = landmarks.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path} is missing non-empty landmark tier {key}")
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected shape for {key} in {path}: {arr.shape}")
        frame_count = arr.shape[0] if frame_count is None else min(frame_count, arr.shape[0])
        arrays.append(arr)
    return np.concatenate([a[:frame_count] for a in arrays], axis=1)


def iter_segments(root: str | Path, include_face: bool = False) -> Iterable[dict]:
    skipped = 0
    for path in sorted(json_dir(root).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        fps = float(data.get("metadata", {}).get("video_fps", 30.0))
        try:
            motion = load_keypoints(path, include_face)
        except ValueError as exc:
            # The released data may contain both *_keypoints_2d and
            # *_keypoints_3d JSONs.  Do not mix coordinate systems silently.
            skipped += 1
            continue
        for index, item in enumerate(data.get("sign_script", {}).get("sign_gestures_both", []) or []):
            gloss = item.get("gloss_id")
            if not gloss:
                continue
            start = float(item.get("start", item.get("begin", 0)))
            end = float(item.get("end", item.get("stop", start)))
            lo = max(0, min(len(motion), round(start * fps)))
            hi = max(lo + 1, min(len(motion), round(end * fps)))
            yield {"id": path.stem, "segment_index": index, "gloss": str(gloss),
                   "fps": fps, "motion": motion[lo:hi]}
    if skipped:
        warnings.warn(
            f"Skipped {skipped} JSON files without the requested 3D landmark tiers. "
            "2D and 3D keypoints are intentionally not mixed.",
            RuntimeWarning,
        )


def compute_stats(root: str | Path, include_face: bool = False) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_x = None
    sum_x2 = None
    for row in iter_segments(root, include_face):
        x = row["motion"].reshape(-1, row["motion"].shape[-1]).astype(np.float64)
        sum_x = x.sum(0) if sum_x is None else sum_x + x.sum(0)
        sum_x2 = (x * x).sum(0) if sum_x2 is None else sum_x2 + (x * x).sum(0)
        total += len(x)
    if total == 0:
        raise RuntimeError("No sign segments were found")
    mean = sum_x / total
    std = np.sqrt(np.maximum(sum_x2 / total - mean * mean, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)
