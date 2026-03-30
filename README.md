# Homogenous-WOBJ-calibration

A small Python implementation of the workspace-calibration part of the paper:

> **Automatic Workspace Calibration Using Homography for Pick and Place**

A camera mounted above a robot cell observes a set of known calibration
points. A 3 × 3 homography matrix **H** is estimated from the pixel
coordinates of those points and their corresponding robot-workspace
coordinates. After calibration, any object position detected in the image
can be transformed to a robot-reachable coordinate with a single matrix
multiplication.

---

## How it works

1. **Collect correspondence pairs** – measure *N* ≥ 4 points whose pixel
   coordinates `(u, v)` and workspace coordinates `(X, Y)` are both known.
2. **Estimate H** – the Direct Linear Transform (DLT) algorithm builds a
   2N × 9 design matrix from the correspondences and solves for H via SVD
   (Hartley & Zisserman, §4.4).  Isotropic point normalisation is applied
   for numerical stability.
3. **Transform** – given a new image point, a single homogeneous multiply
   returns the workspace coordinate: `λ [X Y 1]ᵀ = H [u v 1]ᵀ`.

---

## Quick start

```python
from homography_calibration import WorldObjectCalibrator

# ── 1. Calibrate ──────────────────────────────────────────────────────────
cal = WorldObjectCalibrator()

# pixel coordinates of calibration targets
image_pts = [
    [100, 200],
    [400, 200],
    [400, 500],
    [100, 500],
]

# corresponding robot workspace coordinates (e.g. mm)
world_pts = [
    [  0.0,   0.0],
    [300.0,   0.0],
    [300.0, 200.0],
    [  0.0, 200.0],
]

mean_error = cal.calibrate(image_pts, world_pts)
print(f"Calibration reprojection error: {mean_error:.4f} mm")

# ── 2. Transform new detections ───────────────────────────────────────────
u, v = 250, 350          # pixel position of a detected object
X, Y = cal.image_to_world(u, v)
print(f"Robot target: X={X:.2f} mm, Y={Y:.2f} mm")
```

---

## API reference

### `compute_homography(image_points, world_points) → ndarray (3, 3)`

Low-level function. Estimates H from N ≥ 4 point correspondences using
the DLT algorithm with isotropic normalisation.

### `image_to_world(image_point, H) → ndarray (2,)`

Low-level function. Applies homography H to a single pixel coordinate
and returns the corresponding world coordinate.

### `reprojection_error(image_points, world_points, H) → float`

Returns the mean Euclidean distance (in world units) between the
H-projected image points and the reference world points.

### `WorldObjectCalibrator`

High-level class that wraps the above functions.

| Method / property | Description |
|---|---|
| `calibrate(image_pts, world_pts)` | Estimate H; returns mean reprojection error |
| `image_to_world(u, v)` | Map pixel → workspace coordinate |
| `homography` | The estimated 3 × 3 H matrix |
| `is_calibrated` | `True` after a successful `calibrate()` call |

---

## Running the tests

```bash
pip install numpy pytest
python -m pytest test_homography_calibration.py -v
```
