"""Landmark-based image alignment without project, model, or GUI dependencies."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def landmark_array(
    points: Sequence[Sequence[float] | None],
    labels: Sequence[str],
    *,
    name: str = "Pose",
) -> np.ndarray:
    """Validate labeled coordinates, representing obscured points with NaNs."""
    if len(points) != len(labels):
        raise ValueError(
            f"{name} needs all {len(labels)} registration labels resolved; "
            f"found {len(points)}"
        )
    result = np.full((len(labels), 2), np.nan, dtype=np.float32)
    for index, point in enumerate(points):
        if point is None:
            continue
        value = np.asarray(point, dtype=np.float32)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} {labels[index]} is invalid")
        result[index] = value
    visible = result[np.all(np.isfinite(result), axis=1)]
    if len(visible) != len(np.unique(visible, axis=0)):
        raise ValueError(f"{name} contains duplicate registration landmarks")
    return result


def common_landmark_indices(
    source: np.ndarray,
    destination: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Select group landmarks visible in both validated coordinate arrays."""
    configured = np.asarray(indices, dtype=int)
    visible = np.all(np.isfinite(source[configured]), axis=1) & np.all(
        np.isfinite(destination[configured]), axis=1
    )
    return configured[visible]


def fit_landmark_homography(
    source: np.ndarray,
    destination: np.ndarray,
    *,
    allow_affine: bool = False,
    max_error_px: float = 12.0,
    ransac_threshold_px: float = 10.0,
) -> np.ndarray:
    """Map matched source points to destination image coordinates.

    Fit all visible points first and fall back to RANSAC for inconsistent clicks.
    With ``allow_affine``, three points use an affine transform and four points
    use an exact perspective fit. The return value is always a 3x3 matrix.
    """
    source = np.asarray(source, dtype=np.float32)
    destination = np.asarray(destination, dtype=np.float32)
    if (
        source.ndim != 2
        or source.shape[1] != 2
        or source.shape != destination.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(destination))
    ):
        raise ValueError("Expected matching finite arrays of (x, y) coordinates")
    minimum = 3 if allow_affine else 4
    if len(source) < minimum:
        raise ValueError(f"Registration needs at least {minimum} matched landmarks")
    if any(len(np.unique(points, axis=0)) != len(points) for points in (source, destination)):
        raise ValueError("Registration landmarks must have distinct coordinates")
    if any(np.linalg.matrix_rank(points - points[0]) < 2 for points in (source, destination)):
        raise ValueError("Registration landmarks must not all lie on one line")
    if (
        not np.isfinite(max_error_px)
        or max_error_px <= 0
        or not np.isfinite(ransac_threshold_px)
        or ransac_threshold_px <= 0
    ):
        raise ValueError("Registration error thresholds must be positive and finite")

    if allow_affine and len(source) == 3:
        affine = cv2.getAffineTransform(source, destination)
        matrix = np.vstack((affine, (0.0, 0.0, 1.0)))
    elif allow_affine and len(source) == 4:
        matrix = cv2.getPerspectiveTransform(source, destination)
    else:
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is not None:
            projected = cv2.perspectiveTransform(
                source.reshape(1, -1, 2), matrix
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected - destination, axis=1)
        else:
            errors = np.asarray([np.inf])
        if float(errors.max()) > max_error_px:
            matrix, _ = cv2.findHomography(
                source,
                destination,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold_px,
                maxIters=5000,
                confidence=0.995,
            )
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise ValueError("Could not fit a registration homography")
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Invalid registration homography")
    return matrix


def registration_error_diagnostics(
    source: np.ndarray,
    destination: np.ndarray,
    matrix: np.ndarray,
    indices: Sequence[int],
    labels: Sequence[str],
    *,
    inlier_threshold_px: float = 10.0,
) -> dict:
    """Report destination-pixel errors and obscured/outlier labels for a group."""
    configured = np.asarray(indices, dtype=int)
    selected = common_landmark_indices(source, destination, indices)
    obscured = configured[~np.isin(configured, selected)]
    result = {
        "rmse": None,
        "median": None,
        "max": None,
        "inlier_count": 0,
        "landmark_count": len(selected),
        "expected_landmark_count": len(indices),
        "obscured_labels": [labels[index] for index in obscured],
        "outlier_labels": [],
        "errors_by_label": {},
    }
    if not len(selected):
        return result
    projected = cv2.perspectiveTransform(
        source[selected].reshape(1, -1, 2), matrix
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - destination[selected], axis=1)
    inliers = errors <= inlier_threshold_px
    result.update({
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "inlier_count": int(inliers.sum()),
        "outlier_labels": [
            labels[index] for index, is_inlier in zip(selected, inliers) if not is_inlier
        ],
        "errors_by_label": {
            labels[index]: float(error) for index, error in zip(selected, errors)
        },
    })
    return result
