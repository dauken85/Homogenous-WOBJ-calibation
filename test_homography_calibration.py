"""Unit tests for homography_calibration.py"""

import math
import pytest
import numpy as np
from homography_calibration import (
    compute_homography,
    image_to_world,
    reprojection_error,
    WorldObjectCalibrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_H():
    """A simple but non-trivial 3×3 homography (affine + perspective)."""
    H = np.array([
        [1.2,  0.05, 50.0],
        [0.03, 0.9,  30.0],
        [1e-4, 2e-4,  1.0],
    ])
    return H


def _apply_H(H, pts):
    """Apply homography H to an array of 2-D points, returns (N,2) array."""
    pts = np.asarray(pts, dtype=float)
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    res = (H @ pts_h.T).T
    return res[:, :2] / res[:, [2]]


# ---------------------------------------------------------------------------
# compute_homography
# ---------------------------------------------------------------------------

class TestComputeHomography:

    def test_exact_four_points(self):
        """Homography recovered from exactly 4 correspondences should be
        nearly perfect (zero reprojection error)."""
        H_true = _make_synthetic_H()
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)

        H_est = compute_homography(img_pts, wld_pts)

        err = reprojection_error(img_pts, wld_pts, H_est)
        assert err < 1e-6, f"Reprojection error should be ~0, got {err}"

    def test_overdetermined_many_points(self):
        """DLT should still give low error with many correspondences."""
        H_true = _make_synthetic_H()
        rng = np.random.default_rng(42)
        img_pts = rng.uniform(0, 640, size=(20, 2))
        wld_pts = _apply_H(H_true, img_pts)

        H_est = compute_homography(img_pts, wld_pts)
        err = reprojection_error(img_pts, wld_pts, H_est)
        assert err < 1e-4

    def test_raises_on_fewer_than_four_points(self):
        with pytest.raises(ValueError, match="At least 4"):
            compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            compute_homography([[0, 0], [1, 0], [1, 1], [0, 1]],
                               [[0, 0], [1, 0], [1, 1]])

    def test_raises_on_bad_shape_image(self):
        with pytest.raises(ValueError, match="shape"):
            compute_homography([[0, 0, 0]], [[0, 0]])

    def test_raises_on_bad_shape_world(self):
        with pytest.raises(ValueError, match="shape"):
            compute_homography([[0, 0]], [[0, 0, 0]])

    def test_returns_normalized_matrix(self):
        """H[2,2] should be 1.0 after normalisation."""
        H_true = _make_synthetic_H()
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)

        H_est = compute_homography(img_pts, wld_pts)
        assert abs(H_est[2, 2] - 1.0) < 1e-10

    def test_pure_translation(self):
        """Simple translation homography."""
        tx, ty = 100.0, 200.0
        H_true = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
        img_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)

        H_est = compute_homography(img_pts, wld_pts)
        err = reprojection_error(img_pts, wld_pts, H_est)
        assert err < 1e-6

    def test_pure_scaling(self):
        """Uniform scaling homography."""
        s = 3.5
        H_true = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=float)
        img_pts = np.array([[10, 10], [100, 10], [100, 80], [10, 80]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)

        H_est = compute_homography(img_pts, wld_pts)
        err = reprojection_error(img_pts, wld_pts, H_est)
        assert err < 1e-6

    def test_accepts_list_input(self):
        """Accepts plain Python lists without conversion errors."""
        H_true = _make_synthetic_H()
        img_pts = [[10, 20], [300, 25], [280, 400], [15, 390]]
        wld_pts = _apply_H(H_true, np.array(img_pts)).tolist()
        H_est = compute_homography(img_pts, wld_pts)
        assert H_est.shape == (3, 3)


# ---------------------------------------------------------------------------
# image_to_world
# ---------------------------------------------------------------------------

class TestImageToWorld:

    def test_identity_homography(self):
        H = np.eye(3)
        result = image_to_world([120.0, 340.0], H)
        np.testing.assert_allclose(result, [120.0, 340.0], atol=1e-10)

    def test_translation_homography(self):
        tx, ty = 50.0, -30.0
        H = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
        result = image_to_world([100.0, 200.0], H)
        np.testing.assert_allclose(result, [150.0, 170.0], atol=1e-10)

    def test_roundtrip_with_computed_H(self):
        H_true = _make_synthetic_H()
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)

        H_est = compute_homography(img_pts, wld_pts)
        test_pt = np.array([150.0, 200.0])
        wld_expected = _apply_H(H_true, test_pt[None])[0]
        wld_got = image_to_world(test_pt, H_est)
        np.testing.assert_allclose(wld_got, wld_expected, atol=1e-5)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="length 2"):
            image_to_world([1, 2, 3], np.eye(3))

    def test_returns_ndarray(self):
        result = image_to_world([0.0, 0.0], np.eye(3))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# reprojection_error
# ---------------------------------------------------------------------------

class TestReprojectionError:

    def test_zero_error_perfect_H(self):
        H_true = _make_synthetic_H()
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)
        err = reprojection_error(img_pts, wld_pts, H_true)
        assert err < 1e-10

    def test_nonzero_error_wrong_H(self):
        H_true = _make_synthetic_H()
        H_wrong = np.eye(3)
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)
        err = reprojection_error(img_pts, wld_pts, H_wrong)
        assert err > 1.0


# ---------------------------------------------------------------------------
# WorldObjectCalibrator
# ---------------------------------------------------------------------------

class TestWorldObjectCalibrator:

    def _setup_calibrator(self):
        H_true = _make_synthetic_H()
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390],
                             [160, 210], [240, 180]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)
        cal = WorldObjectCalibrator()
        err = cal.calibrate(img_pts, wld_pts)
        return cal, H_true, err

    def test_calibrate_returns_low_error(self):
        _, _, err = self._setup_calibrator()
        assert err < 1e-4

    def test_is_calibrated_flag(self):
        cal = WorldObjectCalibrator()
        assert not cal.is_calibrated
        cal.calibrate([[0, 0], [1, 0], [1, 1], [0, 1]],
                      [[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cal.is_calibrated

    def test_homography_property_raises_before_calibration(self):
        cal = WorldObjectCalibrator()
        with pytest.raises(RuntimeError, match="calibrate"):
            _ = cal.homography

    def test_image_to_world_accuracy(self):
        cal, H_true, _ = self._setup_calibrator()
        u, v = 180.0, 250.0
        expected = _apply_H(H_true, np.array([[u, v]]))[0]
        X, Y = cal.image_to_world(u, v)
        assert math.isclose(X, expected[0], rel_tol=1e-5)
        assert math.isclose(Y, expected[1], rel_tol=1e-5)

    def test_image_to_world_returns_floats(self):
        cal, _, _ = self._setup_calibrator()
        X, Y = cal.image_to_world(100, 200)
        assert isinstance(X, float)
        assert isinstance(Y, float)

    def test_homography_shape(self):
        cal, _, _ = self._setup_calibrator()
        assert cal.homography.shape == (3, 3)

    def test_recalibrate(self):
        """Calling calibrate() again replaces the previous homography."""
        cal = WorldObjectCalibrator()
        # First calibration
        cal.calibrate([[0, 0], [1, 0], [1, 1], [0, 1]],
                      [[0, 0], [1, 0], [1, 1], [0, 1]])
        H1 = cal.homography.copy()

        # Second calibration with different data
        H_true = _make_synthetic_H()
        img_pts = np.array([[10, 20], [300, 25], [280, 400], [15, 390]], dtype=float)
        wld_pts = _apply_H(H_true, img_pts)
        cal.calibrate(img_pts, wld_pts)

        assert not np.allclose(H1, cal.homography)

    def test_calibrate_with_lists(self):
        """Accepts plain Python list inputs."""
        image_pts = [[100, 200], [400, 200], [400, 500], [100, 500]]
        world_pts = [[0.0, 0.0], [300.0, 0.0], [300.0, 200.0], [0.0, 200.0]]
        cal = WorldObjectCalibrator()
        err = cal.calibrate(image_pts, world_pts)
        assert err < 1e-6
        assert cal.is_calibrated
