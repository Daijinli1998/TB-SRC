"""Cross-view compensation for complementary stereo occlusion.

The legacy scripts in this project triangulate a keypoint only when it is
visible in both cameras.  This module handles the complementary case: exactly
one camera observes a joint.  The visible pixel defines a 3-D camera ray; a
causal constant-velocity prior and an online bone-length estimate disambiguate
depth on that ray.  The recovered 3-D point is then projected into the
occluded camera.

The implementation deliberately depends only on NumPy at runtime.  OpenCV is
not required for the geometry, and PyYAML is optional (a small parser handles
the OpenCV-style calibration files already stored in this repository).

Typical offline use::

    python stereo_occlusion_compensation.py \
      --left recordings_analysis/SESSION/keypoints_2d_left.csv \
      --right recordings_analysis/SESSION/keypoints_2d_right.csv \
      --camera-config configs/camera_config_pose_640x480.yaml \
      --output-dir compensated/SESSION

Important evidence boundary: a compensated coordinate is a model estimate,
not ground truth.  Validate it with held-out visible frames under synthetic
single-view masking, and with independent 3-D truth when available.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


JOINT_ORDER: Tuple[str, ...] = (
    "lhip",
    "rhip",
    "hip",
    "lkne",
    "rkne",
    "lank",
    "rank",
    "ltoe",
    "rtoe",
)

PARENT: Dict[str, str] = {
    "lkne": "lhip",
    "rkne": "rhip",
    "lank": "lkne",
    "rank": "rkne",
    "ltoe": "lank",
    "rtoe": "rank",
}

DEFAULT_INDEX: Dict[str, int] = {
    "lhip": 11,
    "rhip": 12,
    "lkne": 13,
    "rkne": 14,
    "lank": 15,
    "rank": 16,
    "hip": 19,
    "ltoe": 20,
    "rtoe": 21,
}


@dataclass(frozen=True)
class Observation:
    """One 2-D keypoint observation."""

    x: float
    y: float
    confidence: float
    kpt_idx: int = -1

    def is_visible(self, threshold: float) -> bool:
        return (
            math.isfinite(self.x)
            and math.isfinite(self.y)
            and math.isfinite(self.confidence)
            and self.confidence >= threshold
        )


@dataclass
class JointEstimate:
    """Resolved state for one joint in one frame."""

    point_3d: Optional[np.ndarray]
    left_xy: Optional[np.ndarray]
    right_xy: Optional[np.ndarray]
    status: str
    source_view: str
    left_visible: bool
    right_visible: bool
    left_confidence: float
    right_confidence: float


def _numbers_from_yaml_block(text: str, key: str) -> Sequence[float]:
    """Extract ``data:`` values from one top-level OpenCV YAML block."""

    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == f"{key}:" and len(line) - len(line.lstrip()) == 0:
            start = idx + 1
            break
    if start is None:
        raise KeyError(f"Calibration key not found: {key}")

    values = []
    in_data = False
    for line in lines[start:]:
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped and indent == 0:
            break
        if stripped == "data:":
            in_data = True
            continue
        if in_data:
            match = re.match(
                r"^\s*-\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
                line,
            )
            if match:
                values.append(float(match.group(1)))
            elif stripped and not stripped.startswith("-"):
                break
    if not values:
        raise ValueError(f"Calibration block {key!r} has no data values")
    return values


def _load_calibration_mapping(path: os.PathLike[str] | str) -> Mapping[str, object]:
    """Load YAML with PyYAML when present, otherwise parse required blocks."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, Mapping):
            return loaded
    except ImportError:
        pass

    return {
        key: {"data": _numbers_from_yaml_block(text, key)}
        for key in (
            "R",
            "T",
            "camera_matrix_left",
            "camera_matrix_right",
            "dist_coeff_left",
            "dist_coeff_right",
        )
    }


def _matrix(mapping: Mapping[str, object], key: str, shape: Tuple[int, ...]) -> np.ndarray:
    block = mapping[key]
    if isinstance(block, Mapping):
        values = block.get("data")
    else:
        values = block
    array = np.asarray(values, dtype=np.float64)
    expected = int(np.prod(shape))
    if array.size < expected:
        raise ValueError(f"Calibration {key} needs {expected} values, found {array.size}")
    return array[:expected].reshape(shape)


@dataclass(frozen=True)
class StereoCalibration:
    """Unrectified stereo model with points expressed in the left camera frame.

    The repository calibration follows OpenCV's convention::

        X_right = R @ X_left + T
    """

    left_k: np.ndarray
    right_k: np.ndarray
    left_dist: np.ndarray
    right_dist: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_yaml(cls, path: os.PathLike[str] | str) -> "StereoCalibration":
        data = _load_calibration_mapping(path)
        return cls(
            left_k=_matrix(data, "camera_matrix_left", (3, 3)),
            right_k=_matrix(data, "camera_matrix_right", (3, 3)),
            left_dist=_matrix(data, "dist_coeff_left", (5,)),
            right_dist=_matrix(data, "dist_coeff_right", (5,)),
            rotation=_matrix(data, "R", (3, 3)),
            translation=_matrix(data, "T", (3,)),
        )

    @staticmethod
    def _undistort_normalized(
        pixel: Sequence[float],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        iterations: int = 8,
    ) -> np.ndarray:
        """Invert the five-parameter Brown-Conrady distortion model."""

        fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
        cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
        xd = (float(pixel[0]) - cx) / fx
        yd = (float(pixel[1]) - cy) / fy
        x, y = xd, yd
        k1, k2, p1, p2, k3 = [float(v) for v in distortion[:5]]
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            if abs(radial) < 1e-12:
                break
            delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - delta_x) / radial
            y = (yd - delta_y) / radial
        return np.asarray([x, y], dtype=np.float64)

    @staticmethod
    def _distort_normalized(
        normalized: Sequence[float], distortion: np.ndarray
    ) -> np.ndarray:
        x, y = float(normalized[0]), float(normalized[1])
        k1, k2, p1, p2, k3 = [float(v) for v in distortion[:5]]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.asarray([xd, yd], dtype=np.float64)

    def camera_point(self, point_left: Sequence[float], view: str) -> np.ndarray:
        point = np.asarray(point_left, dtype=np.float64).reshape(3)
        if view == "left":
            return point
        if view == "right":
            return self.rotation @ point + self.translation
        raise ValueError("view must be 'left' or 'right'")

    def project(self, point_left: Sequence[float], view: str) -> Optional[np.ndarray]:
        point_camera = self.camera_point(point_left, view)
        z = float(point_camera[2])
        if not math.isfinite(z) or z <= 1e-9:
            return None
        normalized = point_camera[:2] / z
        if view == "left":
            distorted = self._distort_normalized(normalized, self.left_dist)
            k = self.left_k
        elif view == "right":
            distorted = self._distort_normalized(normalized, self.right_dist)
            k = self.right_k
        else:
            raise ValueError("view must be 'left' or 'right'")
        return np.asarray(
            [k[0, 0] * distorted[0] + k[0, 2], k[1, 1] * distorted[1] + k[1, 2]],
            dtype=np.float64,
        )

    def ray(self, pixel: Sequence[float], view: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ray origin/direction in the left camera coordinate frame."""

        if view == "left":
            normalized = self._undistort_normalized(pixel, self.left_k, self.left_dist)
            origin = np.zeros(3, dtype=np.float64)
            direction = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
        elif view == "right":
            normalized = self._undistort_normalized(pixel, self.right_k, self.right_dist)
            ray_right = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
            origin = -self.rotation.T @ self.translation
            direction = self.rotation.T @ ray_right
        else:
            raise ValueError("view must be 'left' or 'right'")
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("Degenerate camera ray")
        return origin, direction / norm

    def triangulate(
        self, left_pixel: Sequence[float], right_pixel: Sequence[float]
    ) -> Optional[np.ndarray]:
        """Linear triangulation using undistorted normalized image points."""

        left = self._undistort_normalized(left_pixel, self.left_k, self.left_dist)
        right = self._undistort_normalized(right_pixel, self.right_k, self.right_dist)
        p_left = np.hstack([np.eye(3), np.zeros((3, 1))])
        p_right = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        a = np.vstack(
            [
                left[0] * p_left[2] - p_left[0],
                left[1] * p_left[2] - p_left[1],
                right[0] * p_right[2] - p_right[0],
                right[1] * p_right[2] - p_right[1],
            ]
        )
        _, _, vh = np.linalg.svd(a)
        homogeneous = vh[-1]
        if abs(float(homogeneous[3])) <= 1e-12:
            return None
        point = homogeneous[:3] / homogeneous[3]
        if not self.valid_point(point):
            return None
        return point.astype(np.float64)

    def valid_point(
        self, point_left: Sequence[float], min_depth: float = 50.0, max_depth: float = 5000.0
    ) -> bool:
        point = np.asarray(point_left, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point)):
            return False
        left_z = float(point[2])
        right_z = float((self.rotation @ point + self.translation)[2])
        return min_depth <= left_z <= max_depth and min_depth <= right_z <= max_depth

    def reprojection_error(
        self,
        point_left: Sequence[float],
        left_pixel: Optional[Sequence[float]],
        right_pixel: Optional[Sequence[float]],
    ) -> float:
        errors = []
        for view, observed in (("left", left_pixel), ("right", right_pixel)):
            if observed is None:
                continue
            projected = self.project(point_left, view)
            if projected is not None:
                errors.append(float(np.linalg.norm(projected - np.asarray(observed, dtype=np.float64))))
        return float(np.mean(errors)) if errors else math.nan


class StereoOcclusionCompensator:
    """Causal joint-level stereo visibility selector and depth resolver."""

    def __init__(
        self,
        calibration: StereoCalibration,
        confidence_threshold: float = 0.3,
        use_bone_constraint: bool = True,
        min_depth: float = 50.0,
        max_depth: float = 5000.0,
        temporal_scale_mm: float = 120.0,
        bone_scale_mm: float = 25.0,
        bone_update_rate: float = 0.05,
    ) -> None:
        self.calibration = calibration
        self.confidence_threshold = float(confidence_threshold)
        self.use_bone_constraint = bool(use_bone_constraint)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.temporal_scale_mm = float(temporal_scale_mm)
        self.bone_scale_mm = float(bone_scale_mm)
        self.bone_update_rate = float(bone_update_rate)
        self.history: Dict[str, list[np.ndarray]] = {name: [] for name in JOINT_ORDER}
        self.bone_lengths: Dict[Tuple[str, str], float] = {}
        self.step_index = -1
        self.pixel_history: Dict[str, Dict[str, list[Tuple[int, np.ndarray]]]] = {
            name: {"left": [], "right": []} for name in JOINT_ORDER
        }
        self.missing_run: Dict[str, Dict[str, int]] = {
            name: {"left": 0, "right": 0} for name in JOINT_ORDER
        }

    def reset(self) -> None:
        self.history = {name: [] for name in JOINT_ORDER}
        self.bone_lengths = {}
        self.step_index = -1
        self.pixel_history = {
            name: {"left": [], "right": []} for name in JOINT_ORDER
        }
        self.missing_run = {
            name: {"left": 0, "right": 0} for name in JOINT_ORDER
        }

    def temporal_prior(self, joint: str) -> Optional[np.ndarray]:
        history = self.history.get(joint, [])
        if len(history) >= 3:
            # A one-step linear fit over several stereo estimates is much less
            # sensitive to disparity jitter than differencing only two frames.
            # During an occlusion the recovered states remain causal members of
            # this history, so the same rule can continue online.
            recent = np.asarray(history[-8:], dtype=np.float64)
            time = np.arange(recent.shape[0], dtype=np.float64)
            design = np.column_stack([time, np.ones_like(time)])
            coefficients, _, _, _ = np.linalg.lstsq(design, recent, rcond=None)
            return np.asarray([recent.shape[0], 1.0]) @ coefficients
        if len(history) == 2:
            return history[-1] + (history[-1] - history[-2])
        if history:
            return history[-1].copy()
        return None

    def _remember(self, joint: str, point: np.ndarray) -> None:
        history = self.history.setdefault(joint, [])
        history.append(np.asarray(point, dtype=np.float64).reshape(3).copy())
        if len(history) > 8:
            del history[:-8]

    def _remember_pixel(self, joint: str, view: str, observation: Observation) -> None:
        history = self.pixel_history[joint][view]
        history.append(
            (
                self.step_index,
                np.asarray([observation.x, observation.y], dtype=np.float64),
            )
        )
        if len(history) > 8:
            del history[:-8]

    def _pixel_prior(self, joint: str, view: str) -> Optional[np.ndarray]:
        """Causal 2-D motion prior in the currently occluded target view."""

        history = self.pixel_history[joint][view]
        if not history:
            return None
        if len(history) == 1:
            return history[-1][1].copy()
        (time_a, point_a), (time_b, point_b) = history[-2:]
        delta = max(1, time_b - time_a)
        velocity = (point_b - point_a) / float(delta)
        return point_b + velocity * float(self.step_index - time_b)

    def _update_bone(self, child: str, parent: str, point: np.ndarray, parent_point: np.ndarray) -> None:
        length = float(np.linalg.norm(point - parent_point))
        if not math.isfinite(length) or length < 20.0 or length > 1000.0:
            return
        key = (parent, child)
        current = self.bone_lengths.get(key)
        if current is None:
            self.bone_lengths[key] = length
            return
        if 0.65 * current <= length <= 1.35 * current:
            rate = self.bone_update_rate
            self.bone_lengths[key] = (1.0 - rate) * current + rate * length

    def _valid_candidate(self, point: np.ndarray) -> bool:
        return self.calibration.valid_point(point, self.min_depth, self.max_depth)

    def _line_sphere_candidates(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> list[np.ndarray]:
        offset = origin - center
        b = 2.0 * float(np.dot(direction, offset))
        c = float(np.dot(offset, offset) - radius * radius)
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            return []
        root = math.sqrt(max(0.0, discriminant))
        candidates = []
        for distance in ((-b - root) / 2.0, (-b + root) / 2.0):
            if distance <= 0.0:
                continue
            point = origin + distance * direction
            if self._valid_candidate(point):
                candidates.append(point)
        return candidates

    def _search_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        prior: Optional[np.ndarray],
        parent_point: Optional[np.ndarray],
        bone_length: Optional[float],
    ) -> Optional[np.ndarray]:
        """Fallback scalar search when noisy observations prevent exact intersection."""

        if prior is not None:
            center_distance = float(np.dot(prior - origin, direction))
        elif parent_point is not None:
            center_distance = float(np.dot(parent_point - origin, direction))
        else:
            return None
        center_distance = max(self.min_depth, min(self.max_depth, center_distance))
        half_width = max(600.0, 0.75 * abs(center_distance))
        low = max(self.min_depth, center_distance - half_width)
        high = min(self.max_depth, center_distance + half_width)
        if high <= low:
            return None

        best_point = None
        for _ in range(4):
            distances = np.linspace(low, high, 257)
            points = origin[None, :] + distances[:, None] * direction[None, :]
            objective = np.zeros(distances.shape, dtype=np.float64)
            if prior is not None:
                temporal = np.linalg.norm(points - prior[None, :], axis=1)
                objective += (temporal / max(self.temporal_scale_mm, 1e-6)) ** 2
            if parent_point is not None and bone_length is not None:
                residual = np.linalg.norm(points - parent_point[None, :], axis=1) - bone_length
                objective += (residual / max(self.bone_scale_mm, 1e-6)) ** 2
            valid = np.asarray([self._valid_candidate(point) for point in points], dtype=bool)
            objective[~valid] = np.inf
            best_index = int(np.argmin(objective))
            if not math.isfinite(float(objective[best_index])):
                return None
            best_point = points[best_index]
            step = (high - low) / 256.0
            low = max(self.min_depth, float(distances[best_index] - 2.0 * step))
            high = min(self.max_depth, float(distances[best_index] + 2.0 * step))
        return None if best_point is None else best_point.astype(np.float64)

    def recover_from_single_view(
        self,
        joint: str,
        pixel: Sequence[float],
        source_view: str,
        parent_point: Optional[np.ndarray],
        target_pixel_prior: Optional[np.ndarray] = None,
        target_missing_run: int = 1,
    ) -> Optional[np.ndarray]:
        origin, direction = self.calibration.ray(pixel, source_view)
        prior = self.temporal_prior(joint)
        if target_pixel_prior is not None:
            if source_view == "left":
                cross_view_prior = self.calibration.triangulate(pixel, target_pixel_prior)
            else:
                cross_view_prior = self.calibration.triangulate(target_pixel_prior, pixel)
            if cross_view_prior is not None:
                # A recent target-view 2-D extrapolation is very informative at
                # occlusion onset, but its weight must decay as the gap grows.
                pixel_weight = max(
                    0.10,
                    math.exp(-max(0, int(target_missing_run) - 1) / 8.0),
                )
                if prior is None:
                    prior = cross_view_prior
                else:
                    prior = pixel_weight * cross_view_prior + (1.0 - pixel_weight) * prior
        bone_length = None
        parent_name = PARENT.get(joint)
        if self.use_bone_constraint and parent_name is not None:
            bone_length = self.bone_lengths.get((parent_name, joint))

        if parent_point is not None and bone_length is not None:
            candidates = self._line_sphere_candidates(
                origin, direction, parent_point, bone_length
            )
            # With a temporal prior, treat bone length as a soft constraint.
            # Enforcing the noisy online length as an exact sphere can move the
            # point tens of millimetres in depth.  Exact intersections are used
            # only for cold-start disambiguation; otherwise the scalar search
            # balances motion and structure.
            if candidates and prior is None:
                reference = prior if prior is not None else parent_point
                return min(candidates, key=lambda point: float(np.linalg.norm(point - reference)))

        return self._search_ray(origin, direction, prior, parent_point, bone_length)

    def step(
        self,
        left: Mapping[str, Observation],
        right: Mapping[str, Observation],
        frame_index: Optional[int] = None,
    ) -> Dict[str, JointEstimate]:
        self.step_index = self.step_index + 1 if frame_index is None else int(frame_index)
        resolved: Dict[str, np.ndarray] = {}
        statuses: Dict[str, str] = {}
        estimates: Dict[str, JointEstimate] = {}

        for joint in JOINT_ORDER:
            obs_left = left.get(joint)
            obs_right = right.get(joint)
            visible_left = obs_left is not None and obs_left.is_visible(self.confidence_threshold)
            visible_right = obs_right is not None and obs_right.is_visible(self.confidence_threshold)
            self.missing_run[joint]["left"] = (
                0 if visible_left else self.missing_run[joint]["left"] + 1
            )
            self.missing_run[joint]["right"] = (
                0 if visible_right else self.missing_run[joint]["right"] + 1
            )
            point = None
            status = "unresolved"
            source_view = "none"

            if visible_left and visible_right:
                point = self.calibration.triangulate(
                    (obs_left.x, obs_left.y), (obs_right.x, obs_right.y)
                )
                if point is not None:
                    status = "stereo"
                    source_view = "both"

            if point is None and (visible_left ^ visible_right):
                if visible_left:
                    source_view = "left"
                    source = obs_left
                    target_view = "right"
                else:
                    source_view = "right"
                    source = obs_right
                    target_view = "left"
                parent_point = resolved.get(PARENT.get(joint, ""))
                point = self.recover_from_single_view(
                    joint,
                    (source.x, source.y),
                    source_view,
                    parent_point,
                    target_pixel_prior=self._pixel_prior(joint, target_view),
                    target_missing_run=self.missing_run[joint][target_view],
                )
                status = f"single_{source_view}" if point is not None else "unresolved_no_depth_prior"

            if point is None and visible_left and visible_right:
                # A failed triangulation is usually a geometry/outlier problem.  Use
                # the higher-confidence view only if history can disambiguate depth.
                if obs_left.confidence >= obs_right.confidence:
                    source_view, source = "left", obs_left
                else:
                    source_view, source = "right", obs_right
                parent_point = resolved.get(PARENT.get(joint, ""))
                point = self.recover_from_single_view(
                    joint,
                    (source.x, source.y),
                    source_view,
                    parent_point,
                )
                status = f"single_{source_view}_fallback" if point is not None else "unresolved_bad_geometry"

            left_xy = self.calibration.project(point, "left") if point is not None else None
            right_xy = self.calibration.project(point, "right") if point is not None else None
            if point is not None:
                resolved[joint] = point
                statuses[joint] = status
                self._remember(joint, point)
                parent_name = PARENT.get(joint)
                if (
                    status == "stereo"
                    and parent_name is not None
                    and parent_name in resolved
                    and statuses.get(parent_name) == "stereo"
                ):
                    self._update_bone(joint, parent_name, point, resolved[parent_name])

            estimates[joint] = JointEstimate(
                point_3d=point,
                left_xy=left_xy,
                right_xy=right_xy,
                status=status,
                source_view=source_view,
                left_visible=visible_left,
                right_visible=visible_right,
                left_confidence=float(obs_left.confidence) if obs_left is not None else math.nan,
                right_confidence=float(obs_right.confidence) if obs_right is not None else math.nan,
            )
            if visible_left and obs_left is not None:
                self._remember_pixel(joint, "left", obs_left)
            if visible_right and obs_right is not None:
                self._remember_pixel(joint, "right", obs_right)
        return estimates


def load_observation_csv(path: os.PathLike[str] | str) -> Dict[int, Dict[str, Observation]]:
    """Read either repository CSV schema (``conf`` or ``confidence``)."""

    frames: Dict[int, Dict[str, Observation]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        confidence_field = "conf" if "conf" in reader.fieldnames else "confidence"
        if confidence_field not in reader.fieldnames:
            raise ValueError(f"CSV needs a conf/confidence column: {path}")
        required = {"frame", "kpt_name", "x", "y", confidence_field}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV {path} missing columns: {sorted(missing)}")
        for row in reader:
            try:
                frame = int(float(row["frame"]))
                name = row["kpt_name"].strip()
                observation = Observation(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    confidence=float(row[confidence_field]),
                    kpt_idx=int(float(row.get("kpt_idx", DEFAULT_INDEX.get(name, -1)))),
                )
            except (TypeError, ValueError):
                continue
            frames.setdefault(frame, {})[name] = observation
    return frames


def _write_corrected_rows(
    writer: csv.writer,
    frame: int,
    view: str,
    observations: Mapping[str, Observation],
    estimates: Mapping[str, JointEstimate],
) -> None:
    for name in JOINT_ORDER:
        estimate = estimates[name]
        observed = observations.get(name)
        visible = estimate.left_visible if view == "left" else estimate.right_visible
        projected = estimate.left_xy if view == "left" else estimate.right_xy
        if visible and observed is not None:
            xy = np.asarray([observed.x, observed.y], dtype=np.float64)
            detector_confidence: object = observed.confidence
            support_confidence: object = observed.confidence
            origin = "observed"
        elif projected is not None:
            xy = projected
            source_confidence = (
                estimate.left_confidence if estimate.source_view == "left" else estimate.right_confidence
            )
            # The source-view detector score is supporting evidence; it is not
            # a calibrated confidence for the synthesized target coordinate.
            detector_confidence = ""
            support_confidence = source_confidence if math.isfinite(source_confidence) else ""
            origin = "compensated"
        else:
            continue
        writer.writerow(
            [
                frame,
                observed.kpt_idx if observed is not None else DEFAULT_INDEX[name],
                name,
                float(xy[0]),
                float(xy[1]),
                detector_confidence,
                support_confidence,
                origin,
                estimate.status,
                estimate.source_view,
                "" if observed is None else float(observed.confidence),
            ]
        )


def run_csv_compensation(
    left_csv: os.PathLike[str] | str,
    right_csv: os.PathLike[str] | str,
    calibration_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    confidence_threshold: float = 0.3,
    use_bone_constraint: bool = True,
) -> Dict[str, Path]:
    left_frames = load_observation_csv(left_csv)
    right_frames = load_observation_csv(right_csv)
    calibration = StereoCalibration.from_yaml(calibration_path)
    compensator = StereoOcclusionCompensator(
        calibration,
        confidence_threshold=confidence_threshold,
        use_bone_constraint=use_bone_constraint,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    left_path = output / "keypoints_2d_left_compensated.csv"
    right_path = output / "keypoints_2d_right_compensated.csv"
    three_d_path = output / "keypoints_3d_compensated.csv"

    two_d_header = [
        "frame",
        "kpt_idx",
        "kpt_name",
        "x",
        "y",
        "conf",
        "support_conf",
        "origin",
        "status",
        "source_view",
        "original_conf",
    ]
    three_d_header = [
        "frame",
        "kpt_idx",
        "kpt_name",
        "X",
        "Y",
        "Z",
        "status",
        "source_view",
        "left_visible",
        "right_visible",
        "conf_l",
        "conf_r",
    ]

    frames = sorted(set(left_frames).union(right_frames))
    with (
        left_path.open("w", encoding="utf-8", newline="") as left_handle,
        right_path.open("w", encoding="utf-8", newline="") as right_handle,
        three_d_path.open("w", encoding="utf-8", newline="") as three_d_handle,
    ):
        left_writer = csv.writer(left_handle)
        right_writer = csv.writer(right_handle)
        three_d_writer = csv.writer(three_d_handle)
        left_writer.writerow(two_d_header)
        right_writer.writerow(two_d_header)
        three_d_writer.writerow(three_d_header)
        for frame in frames:
            left_obs = left_frames.get(frame, {})
            right_obs = right_frames.get(frame, {})
            estimates = compensator.step(left_obs, right_obs, frame_index=frame)
            _write_corrected_rows(left_writer, frame, "left", left_obs, estimates)
            _write_corrected_rows(right_writer, frame, "right", right_obs, estimates)
            for name in JOINT_ORDER:
                estimate = estimates[name]
                if estimate.point_3d is None:
                    continue
                observed = left_obs.get(name) or right_obs.get(name)
                index = observed.kpt_idx if observed is not None else DEFAULT_INDEX[name]
                point = estimate.point_3d
                three_d_writer.writerow(
                    [
                        frame,
                        index,
                        name,
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        estimate.status,
                        estimate.source_view,
                        int(estimate.left_visible),
                        int(estimate.right_visible),
                        estimate.left_confidence,
                        estimate.right_confidence,
                    ]
                )
    return {"left": left_path, "right": right_path, "three_d": three_d_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compensate complementary stereo keypoint occlusion from paired 2-D CSV files."
    )
    parser.add_argument("--left", required=True, help="Left-view 2-D keypoint CSV")
    parser.add_argument("--right", required=True, help="Right-view 2-D keypoint CSV")
    parser.add_argument("--camera-config", required=True, help="Stereo calibration YAML")
    parser.add_argument("--output-dir", required=True, help="Directory for corrected CSV files")
    parser.add_argument("--conf-thr", type=float, default=0.3, help="Visibility threshold")
    parser.add_argument(
        "--no-bone-constraint",
        action="store_true",
        help="Use the temporal ray prior without the parent-child bone constraint",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_csv_compensation(
        args.left,
        args.right,
        args.camera_config,
        args.output_dir,
        confidence_threshold=args.conf_thr,
        use_bone_constraint=not args.no_bone_constraint,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
