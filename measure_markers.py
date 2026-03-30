"""
Measure marker positions relative to marker 0 using solvePnP.

Detects ArUco markers, estimates each marker's corner-0 position in
camera space via solvePnP, then transforms all positions so that
marker 0 is the origin (x=0, y=0). Outputs positions in the same
JSON format as workspace_config.json world_points.

Pipeline:
  1. Capture image (camera or file)
  2. Detect ArUco markers -> get corner pixel coords
  3. solvePnP per marker -> get 3D position (tvec) in camera space
  4. Subtract origin marker tvec -> relative positions
  5. Fit similarity transform using True_value.json to correct
     axis flip / rotation / scale from camera space to workspace
  6. Output corrected world_points JSON
"""

import json
import os
import sys
import argparse

import cv2
import numpy as np

from calibrate_workspace import ARUCO_DICT_MAP, detect_aruco_markers


def estimate_camera_matrix(image_shape):
    """
    Build an approximate camera intrinsic matrix from image dimensions.

    Assumes focal length equals the image width (a rough default when
    no real calibration is available). The principal point is placed
    at the image center.

    Args:
        image_shape: (height, width, ...) from image.shape

    Returns:
        3x3 numpy camera matrix (float64).
    """
    h, w = image_shape[:2]
    f = float(w)  # rough focal length estimate (pixels)
    return np.array([
        [f, 0, w / 2.0],   # fx, 0, cx
        [0, f, h / 2.0],   # 0, fy, cy
        [0, 0, 1],
    ], dtype=np.float64)


def solve_marker_poses(corners_list, ids_flat, marker_size_mm, camera_matrix, dist_coeffs):
    """
    Estimate 3D pose of each marker using cv2.solvePnP.

    For each marker, defines a 3D object model (a square of side
    marker_size_mm with corner 0 at the origin) and solves for the
    rotation + translation that maps it to the detected 2D corners.

    The translation vector (tvec) gives the marker's corner-0 position
    in camera coordinates:
      - tvec[0] = X (right in camera view)
      - tvec[1] = Y (down in camera view)
      - tvec[2] = Z (forward / depth)

    Args:
        corners_list: list of corner arrays from ArUco detection.
        ids_flat: 1D array of marker IDs.
        marker_size_mm: physical side length of each marker (mm).
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: distortion coefficients (5-element array).

    Returns:
        dict {marker_id (int): tvec (3-element numpy array)}
    """
    s = float(marker_size_mm)
    # 3D model of one marker: corner 0 at origin, clockwise
    obj_pts = np.array([
        [0, 0, 0],     # corner 0 (top-left)
        [s, 0, 0],     # corner 1 (top-right)
        [s, s, 0],     # corner 2 (bottom-right)
        [0, s, 0],     # corner 3 (bottom-left)
    ], dtype=np.float64)

    poses = {}
    for corners, mid in zip(corners_list, ids_flat):
        # corners shape: (1, 4, 2) -> reshape to (4, 2) pixel coords
        img_pts = corners.reshape(4, 2).astype(np.float64)

        # solvePnP finds rvec (rotation) and tvec (translation) such that
        # the 3D obj_pts project onto img_pts through the camera model
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        if ok:
            # tvec is (3,1) -> flatten to (3,) for easier math
            poses[int(mid)] = tvec.flatten()
    return poses


def main():
    # =====================================================================
    # 1. Parse arguments
    # =====================================================================
    parser = argparse.ArgumentParser(
        description="Measure marker positions relative to marker 0."
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "workspace_config.json"),
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a color image (skips camera capture).")
    parser.add_argument("--origin", type=int, default=0,
                        help="Marker ID to use as origin (default: 0).")
    parser.add_argument("--true-values", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "True_value.json"),
                        help="Path to JSON with true world points (for axis correction).")
    args = parser.parse_args()

    # =====================================================================
    # 2. Load workspace config (marker dictionary, size, expected positions)
    # =====================================================================
    with open(args.config, "r") as f:
        config = json.load(f)

    dict_name = config["aruco_dictionary"]
    marker_size_mm = config["marker_size_mm"]    # physical side length (mm)
    world_points = config["world_points"]         # expected positions from config
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])

    # =====================================================================
    # 3. Capture or load image
    # =====================================================================
    if args.image:
        color_image = cv2.imread(args.image)
        if color_image is None:
            print(f"[ERROR] Could not load image: {args.image}")
            sys.exit(1)
    else:
        from camera import OrbbecCamera
        print("Capturing frame from camera...")
        with OrbbecCamera() as cam:
            color_image, _ = cam.capture()
        if color_image is None:
            print("[ERROR] Failed to capture frame.")
            sys.exit(1)

    # =====================================================================
    # 4. Detect ArUco markers in the image
    # =====================================================================
    corners_list, ids_flat, centroids = detect_aruco_markers(color_image, aruco_dict)
    print(f"Detected markers: {sorted(centroids.keys())}")

    if len(corners_list) == 0:
        print("[ERROR] No markers detected.")
        sys.exit(1)

    # =====================================================================
    # 5. Estimate each marker's 3D pose via solvePnP
    #    Using estimated intrinsics (no real calibration file available).
    #    tvec gives position in camera coordinate system.
    # =====================================================================
    camera_matrix = estimate_camera_matrix(color_image.shape)
    dist_coeffs = np.zeros(5, dtype=np.float64)  # assume zero distortion
    poses = solve_marker_poses(corners_list, ids_flat, marker_size_mm,
                               camera_matrix, dist_coeffs)

    origin_id = args.origin
    if origin_id not in poses:
        print(f"[ERROR] Origin marker ID {origin_id} not detected.")
        sys.exit(1)

    # =====================================================================
    # 6. Compute relative positions in camera space
    #    Subtract the origin marker's tvec so that it becomes [0, 0].
    #    These are raw camera-axis values (X-right, Y-down, Z-forward).
    #    The affine transform in step 7 will correct axis orientation.
    # =====================================================================
    origin = poses[origin_id]  # tvec of the origin marker
    raw_points = {}
    sorted_ids = sorted(poses.keys())
    for mid in sorted_ids:
        rel = poses[mid] - origin  # relative tvec [dx, dy, dz]
        # Keep only X and Y (ignore Z / depth for planar workspace)
        raw_points[str(mid)] = [float(rel[0]), float(rel[1])]

    print("\nRaw camera-space positions (before correction):")
    for mid_str in sorted(raw_points.keys(), key=int):
        print(f"  ID {mid_str}: {raw_points[mid_str]}")

    # =====================================================================
    # 7. Correct axis alignment using hand-measured true values
    #
    #    Camera axes (X-right, Y-down) don't match workspace axes.
    #    We fit a full 2D affine transform (rotation, independent X/Y
    #    scale, reflection, shear, translation) from raw camera-space
    #    points -> true workspace points. This corrects:
    #      - Y-axis flip (camera Y-down vs workspace Y-up)
    #      - Rotation between camera and workspace
    #      - Non-uniform scale (from inaccurate focal length estimate)
    # =====================================================================
    true_points = {}
    if os.path.exists(args.true_values):
        with open(args.true_values, "r") as f:
            true_points = json.load(f).get("world_points", {})

    # Collect markers that appear in both raw detections and true values
    # (includes origin [0,0]->[0,0] which anchors the transform)
    common_ids = [k for k in raw_points if k in true_points]

    if len(common_ids) >= 2:
        # Source: raw camera-space positions (Nx2)
        src = np.array([raw_points[k] for k in common_ids], dtype=np.float64)
        # Destination: hand-measured true positions (Nx2)
        dst = np.array([true_points[k][:2] for k in common_ids], dtype=np.float64)

        # estimateAffine2D fits a full 2x3 affine: [A | t]
        # where A is a 2x2 matrix (rotation, scale_x, scale_y, shear)
        # and t is translation. This handles axis flip + non-uniform scale.
        T, inliers = cv2.estimateAffine2D(
            src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2)
        )

        if T is not None:
            print(f"\nAxis correction transform (fitted from {len(common_ids)} markers):")
            print(f"  {T}")

            # Apply the 2x3 affine transform to every raw point
            measured_points = {}
            for mid_str in sorted(raw_points.keys(), key=int):
                pt = np.array([[raw_points[mid_str]]], dtype=np.float64)
                corrected = cv2.transform(pt, T).flatten()  # [corrected_x, corrected_y]
                measured_points[mid_str] = [round(corrected[0], 1), round(corrected[1], 1)]

            # Force origin marker to exactly [0.0, 0.0]
            # (remove any small residual offset from the transform)
            origin_key = str(origin_id)
            if origin_key in measured_points:
                offset = measured_points[origin_key]
                for mid_str in measured_points:
                    measured_points[mid_str] = [
                        round(measured_points[mid_str][0] - offset[0], 1),
                        round(measured_points[mid_str][1] - offset[1], 1),
                    ]
        else:
            print("[WARNING] Could not estimate axis correction. Using raw values.")
            measured_points = {k: [round(v[0], 1), round(v[1], 1)] for k, v in raw_points.items()}
    else:
        print("[WARNING] Not enough common markers with true values for axis correction.")
        print("          Using raw camera-space values (may have flipped/rotated axes).")
        measured_points = {k: [round(v[0], 1), round(v[1], 1)] for k, v in raw_points.items()}

    # =====================================================================
    # 8. Output results
    # =====================================================================

    # Print in the same JSON format as workspace_config.json world_points
    print("\nMeasured world_points (relative to marker 0):")
    print(json.dumps({"world_points": measured_points}, indent=4))

    # Print comparison table: measured vs true with per-marker error
    if true_points:
        print("\nComparison with true values:")
        print(f"  {'ID':>4}  {'Meas X':>8} {'Meas Y':>8}  |  {'True X':>8} {'True Y':>8}  |  {'Error':>8}")
        print(f"  {'-'*4}  {'-'*8} {'-'*8}  |  {'-'*8} {'-'*8}  |  {'-'*8}")
        for mid_str in sorted(measured_points.keys(), key=int):
            mx, my = measured_points[mid_str]
            if mid_str in true_points:
                tx, ty = true_points[mid_str][:2]
                err = np.sqrt((mx - tx) ** 2 + (my - ty) ** 2)
                print(f"  {mid_str:>4}  {mx:>8.1f} {my:>8.1f}  |  {tx:>8.1f} {ty:>8.1f}  |  {err:>8.1f}")
            else:
                print(f"  {mid_str:>4}  {mx:>8.1f} {my:>8.1f}  |  {'N/A':>8} {'N/A':>8}  |  {'N/A':>8}")

    # =====================================================================
    # 9. Save results to file
    # =====================================================================
    out_path = os.path.join(os.path.dirname(args.config), "measured_world_points.json")
    with open(out_path, "w") as f:
        json.dump({"world_points": measured_points}, f, indent=4)
    print(f"\nSaved to: {out_path}")

    # =====================================================================
    # 10. Save annotated snapshot
    #     Red dot at each marker's corner 0 with [x, y] label
    # =====================================================================
    annotated = color_image.copy()
    for corners, mid in zip(corners_list, ids_flat):
        mid_int = int(mid)
        if str(mid_int) not in measured_points:
            continue
        # Corner 0 is the first of the 4 detected corners
        c0 = corners.reshape(4, 2)[0]
        cx, cy = int(c0[0]), int(c0[1])
        wx, wy = measured_points[str(mid_int)]

        # Draw red dot at corner 0
        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
        # Label with marker ID and measured world coordinates
        label = f"ID{mid_int} [{wx}, {wy}]"
        cv2.putText(annotated, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 2)

    snap_path = os.path.join(os.path.dirname(args.config), "..", "output", "measured_markers.png")
    os.makedirs(os.path.dirname(snap_path), exist_ok=True)
    cv2.imwrite(snap_path, annotated)
    print(f"Snapshot saved to: {os.path.abspath(snap_path)}")



if __name__ == "__main__":
    main()
