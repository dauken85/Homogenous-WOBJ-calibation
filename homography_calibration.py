"""
Workspace (WOBJ) calibration via homography for pick-and-place robots.

Implements the approach described in:
  "Automatic Workspace Calibration Using Homography for Pick and Place"

A homography H (3×3) is estimated from N ≥ 4 point correspondences
  (image pixel) ↔ (robot-workspace coordinate)
using the Direct Linear Transform (DLT) algorithm with SVD.
Once calibrated, any image point can be mapped to a robot workspace
coordinate with a single matrix multiplication.

Typical usage
-------------
>>> from homography_calibration import WorldObjectCalibrator
>>> cal = WorldObjectCalibrator()
>>> # Collect calibration point pairs (at least 4 required)
>>> image_pts  = [[100, 200], [400, 200], [400, 500], [100, 500]]
>>> world_pts  = [[0.0, 0.0], [300.0, 0.0], [300.0, 200.0], [0.0, 200.0]]
>>> cal.calibrate(image_pts, world_pts)
>>> wx, wy = cal.image_to_world(250, 350)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _normalize_points(pts: np.ndarray):
    """Isotropic normalisation (Hartley & Zisserman, §4.4).

    Translates the centroid to the origin and scales so that the mean
    distance from the origin is √2.

    Parameters
    ----------
    pts : ndarray, shape (N, 2)

    Returns
    -------
    pts_norm : ndarray, shape (N, 2)
    T        : ndarray, shape (3, 3) – normalisation matrix
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    # Avoid division by zero for degenerate configurations
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([
        [scale, 0,     -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,     0,      1                  ],
    ])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_h.T).T[:, :2]
    return pts_norm, T


def compute_homography(image_points: np.ndarray,
                       world_points: np.ndarray) -> np.ndarray:
    """Estimate a 3×3 homography H with the DLT algorithm.

    H maps *image* homogeneous coordinates to *world* homogeneous
    coordinates:  λ · [X, Y, 1]ᵀ = H · [u, v, 1]ᵀ

    Parameters
    ----------
    image_points : array-like, shape (N, 2)   – pixel coordinates (u, v)
    world_points : array-like, shape (N, 2)   – workspace coordinates (X, Y)

    Returns
    -------
    H : ndarray, shape (3, 3)

    Raises
    ------
    ValueError
        If fewer than 4 point pairs are provided or the point arrays have
        different lengths.
    """
    image_points = np.asarray(image_points, dtype=float)
    world_points = np.asarray(world_points, dtype=float)

    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("image_points must have shape (N, 2)")
    if world_points.ndim != 2 or world_points.shape[1] != 2:
        raise ValueError("world_points must have shape (N, 2)")
    if len(image_points) != len(world_points):
        raise ValueError("image_points and world_points must have the same length")
    if len(image_points) < 4:
        raise ValueError("At least 4 point correspondences are required")

    # Normalise for numerical stability
    img_norm, T_img = _normalize_points(image_points)
    wld_norm, T_wld = _normalize_points(world_points)

    # Build the 2N × 9 design matrix A  (one row-pair per correspondence)
    N = len(img_norm)
    A = np.zeros((2 * N, 9))
    for i, ((u, v), (X, Y)) in enumerate(zip(img_norm, wld_norm)):
        A[2 * i]     = [-u, -v, -1,  0,  0,  0, X * u, X * v, X]
        A[2 * i + 1] = [ 0,  0,  0, -u, -v, -1, Y * u, Y * v, Y]

    # Solve using SVD; h is the right singular vector for the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Denormalise: H = T_wld⁻¹ · H_norm · T_img
    H = np.linalg.inv(T_wld) @ H_norm @ T_img
    # Normalise so that H[2, 2] == 1 (removes the scale ambiguity)
    H /= H[2, 2]
    return H


def image_to_world(image_point: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map a single image point to world coordinates using homography H.

    Parameters
    ----------
    image_point : array-like, shape (2,)  – pixel coordinate (u, v)
    H           : ndarray,    shape (3, 3) – homography matrix

    Returns
    -------
    world_point : ndarray, shape (2,)  – workspace coordinate (X, Y)
    """
    pt = np.asarray(image_point, dtype=float)
    if pt.shape != (2,):
        raise ValueError("image_point must be a 1-D array of length 2")
    pt_h = np.array([pt[0], pt[1], 1.0])
    result = H @ pt_h
    if abs(result[2]) < 1e-12:
        raise ValueError("Degenerate homography: w component is zero")
    return result[:2] / result[2]


def reprojection_error(image_points: np.ndarray,
                       world_points: np.ndarray,
                       H: np.ndarray) -> float:
    """Return the mean Euclidean reprojection error (in world units).

    Parameters
    ----------
    image_points : array-like, shape (N, 2)
    world_points : array-like, shape (N, 2)
    H            : ndarray,    shape (3, 3)

    Returns
    -------
    float – mean distance between mapped and reference world points
    """
    image_points = np.asarray(image_points, dtype=float)
    world_points = np.asarray(world_points, dtype=float)
    errors = []
    for img_pt, wld_pt in zip(image_points, world_points):
        estimated = image_to_world(img_pt, H)
        errors.append(np.linalg.norm(estimated - wld_pt))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# High-level calibration class
# ---------------------------------------------------------------------------

class WorldObjectCalibrator:
    """High-level interface for WOBJ (workspace-object) homography calibration.

    Collect calibration point pairs, call :meth:`calibrate`, then use
    :meth:`image_to_world` to transform any subsequent image detections
    into robot-workspace coordinates.
    """

    def __init__(self) -> None:
        self._H: np.ndarray | None = None
        self._image_pts: np.ndarray | None = None
        self._world_pts: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def homography(self) -> np.ndarray:
        """The estimated 3×3 homography matrix (read-only)."""
        if self._H is None:
            raise RuntimeError("Calibrator has not been calibrated yet – "
                               "call calibrate() first")
        return self._H

    @property
    def is_calibrated(self) -> bool:
        """True if :meth:`calibrate` has been called successfully."""
        return self._H is not None

    def calibrate(self,
                  image_points,
                  world_points) -> float:
        """Estimate the homography from point correspondences.

        Parameters
        ----------
        image_points : array-like, shape (N, 2)  – pixel coordinates
        world_points : array-like, shape (N, 2)  – robot workspace coordinates

        Returns
        -------
        float – mean reprojection error (world units) on the calibration set
        """
        self._image_pts = np.asarray(image_points, dtype=float)
        self._world_pts = np.asarray(world_points, dtype=float)
        self._H = compute_homography(self._image_pts, self._world_pts)
        error = reprojection_error(self._image_pts, self._world_pts, self._H)
        return error

    def image_to_world(self, u: float, v: float) -> tuple[float, float]:
        """Transform pixel coordinate (u, v) → workspace coordinate (X, Y).

        Parameters
        ----------
        u : float – column (pixel)
        v : float – row    (pixel)

        Returns
        -------
        (X, Y) : tuple[float, float] – robot workspace coordinate
        """
        result = image_to_world(np.array([u, v]), self.homography)
        return float(result[0]), float(result[1])
