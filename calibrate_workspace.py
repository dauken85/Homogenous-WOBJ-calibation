"""
Workspace calibration using ArUco markers and homography.

Detects ArUco markers in a camera frame, matches them to known world
coordinates, and computes a homography matrix (image -> world transform).
Based on the approach in Burde et al., IEEE CASE 2023.
"""

import json
import os
import sys
import time
import argparse

import cv2
import numpy as np

from camera import get_depth_at_pixel

# Mapping of string names to OpenCV ArUco dictionary constants
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def detect_aruco_markers(color_image, aruco_dict):
    """
    Detect ArUco markers in a BGR image.

    Args:
        color_image: np.ndarray BGR image.
        aruco_dict: OpenCV ArUco dictionary object.

    Returns:
        tuple: (corners_list, ids_flat, centroids_dict)
            - corners_list: list of corner arrays from detectMarkers
            - ids_flat: 1D array of detected marker IDs
            - centroids_dict: dict {str(id): [cx, cy]} pixel centroids
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners_list) == 0:
        return [], np.array([]), {}

    ids_flat = ids.flatten()
    centroids = {}
    for tag_corners, tag_id in zip(corners_list, ids_flat):
        pts = tag_corners.reshape(4, 2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        centroids[str(int(tag_id))] = [cx, cy]

    return corners_list, ids_flat, centroids


def compute_homography(image_points, world_points):
    """
    Compute the homography matrix from image to world coordinates.

    Matches detected marker IDs to their known world positions and
    computes H such that: world_point = H @ image_point (homogeneous).

    Args:
        image_points: dict {str(id): [px_x, px_y]} detected centroids.
        world_points: dict {str(id): [world_x, world_y]} from config.

    Returns:
        tuple: (homography_matrix, matched_image_pts, matched_world_pts, matched_ids)
            homography_matrix is None if fewer than 4 matches.
    """
    matched_img = []
    matched_world = []
    matched_ids = []

    for tag_id, img_pt in image_points.items():
        if tag_id in world_points:
            matched_img.append(img_pt)
            matched_world.append(world_points[tag_id][:2])  # Take only x, y
            matched_ids.append(int(tag_id))

    if len(matched_img) < 4:
        print(f"[WARNING] Only {len(matched_img)} markers matched (need >= 4). "
              "Cannot compute homography.")
        return None, matched_img, matched_world, matched_ids

    img_pts = np.array(matched_img, dtype=np.float32).reshape(-1, 1, 2)
    world_pts = np.array(matched_world, dtype=np.float32).reshape(-1, 1, 2)

    H, status = cv2.findHomography(
        srcPoints=img_pts,
        dstPoints=world_pts,
        method=0,  # Regular least-squares (use cv2.RANSAC for noisy data)
    )

    return H, matched_img, matched_world, matched_ids


def compute_reprojection_error(H, image_points, world_points):
    """
    Compute per-point reprojection error through the homography.

    Transforms image points through H and compares to known world points.

    Args:
        H: 3x3 homography matrix.
        image_points: list of [px_x, px_y].
        world_points: list of [world_x, world_y].

    Returns:
        tuple: (errors_per_point, mean_error)
            errors_per_point: list of L2 distances in mm.
            mean_error: average L2 error in mm.
    """
    img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(img_pts, H).reshape(-1, 2)
    world_pts = np.array(world_points, dtype=np.float32)

    errors = np.linalg.norm(projected - world_pts, axis=1)
    return errors.tolist(), float(np.mean(errors))


def compute_plane_depth(depth_image, image_points, kernel_size=5):
    """
    Compute the average depth of the workspace plane at marker locations.

    Args:
        depth_image: np.ndarray (H, W) depth in mm.
        image_points: list of [px_x, px_y] marker pixel locations.
        kernel_size: Kernel size for depth averaging.

    Returns:
        float: Average plane depth in mm, or 0.0 if no valid depth.
    """
    depths = []
    for pt in image_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        d = get_depth_at_pixel(depth_image, x, y, kernel_size)
        if d > 0:
            depths.append(d)

    if not depths:
        return 0.0
    return float(np.mean(depths))


def draw_calibration_result(color_image, corners_list, ids_flat, centroids,
                            world_points_config, H, errors, matched_ids):
    """
    Draw an annotated calibration visualization.

    Args:
        color_image: BGR image to annotate (will be copied).
        corners_list: Marker corner arrays from detection.
        ids_flat: Detected marker IDs.
        centroids: dict {str(id): [cx, cy]}.
        world_points_config: dict {str(id): [wx, wy]} from config.
        H: Homography matrix (can be None).
        errors: list of per-point errors in mm (parallel to matched_ids).
        matched_ids: list of matched marker IDs.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = color_image.copy()

    # Draw detected marker outlines
    if len(corners_list) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids_flat.reshape(-1, 1))

    # Draw centroid and world coordinate labels
    error_map = {mid: err for mid, err in zip(matched_ids, errors)} if errors else {}

    for tag_id_str, (cx, cy) in centroids.items():
        tag_id = int(tag_id_str)
        ix, iy = int(cx), int(cy)

        # Centroid dot
        cv2.circle(annotated, (ix, iy), 5, (0, 0, 255), -1)

        # World coordinate label
        if tag_id_str in world_points_config:
            wx, wy = world_points_config[tag_id_str][:2]
            label = f"ID{tag_id} [{wx:.0f},{wy:.0f}]mm"
            cv2.putText(annotated, label, (ix + 10, iy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Error label if available
            if tag_id in error_map:
                err_label = f"err: {error_map[tag_id]:.2f}mm"
                cv2.putText(annotated, err_label, (ix + 10, iy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # If homography available, draw a projected grid
    if H is not None:
        H_inv = np.linalg.inv(H)
        # Draw grid lines at 100mm intervals
        grid_range_x = range(0, 901, 100)
        grid_range_y = range(0, 501, 100)

        for gx in grid_range_x:
            pts_world = np.array([[gx, gy] for gy in grid_range_y], dtype=np.float32).reshape(-1, 1, 2)
            pts_img = cv2.perspectiveTransform(pts_world, H_inv).reshape(-1, 2).astype(int)
            for i in range(len(pts_img) - 1):
                cv2.line(annotated, tuple(pts_img[i]), tuple(pts_img[i + 1]),
                         (200, 200, 200), 1)

        for gy in grid_range_y:
            pts_world = np.array([[gx, gy] for gx in grid_range_x], dtype=np.float32).reshape(-1, 1, 2)
            pts_img = cv2.perspectiveTransform(pts_world, H_inv).reshape(-1, 2).astype(int)
            for i in range(len(pts_img) - 1):
                cv2.line(annotated, tuple(pts_img[i]), tuple(pts_img[i + 1]),
                         (200, 200, 200), 1)

    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate workspace using ArUco markers and homography."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "workspace_config.json"),
        help="Path to workspace config JSON.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a color image file (skips camera capture).",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default=None,
        help="Path to a depth image/npy file (used with --image).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "calibration_result.json"),
        help="Path to save calibration result JSON.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated calibration image.",
    )
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        config = json.load(f)

    dict_name = config["aruco_dictionary"]
    world_points_config = config["world_points"]

    if dict_name not in ARUCO_DICT_MAP:
        print(f"[ERROR] Unknown ArUco dictionary: {dict_name}")
        sys.exit(1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])

    # --- Capture or load image ---
    depth_image = None
    if args.image:
        print(f"Loading image from file: {args.image}")
        color_image = cv2.imread(args.image)
        if color_image is None:
            print(f"[ERROR] Could not load image: {args.image}")
            sys.exit(1)
        if args.depth:
            from camera import FileFallbackCamera
            cam = FileFallbackCamera(args.image, args.depth)
            _, depth_image = cam.capture()
    else:
        print("Capturing frame from Orbbec Gemini 2...")
        from camera import OrbbecCamera
        with OrbbecCamera() as cam:
            color_image, depth_image = cam.capture()
        if color_image is None:
            print("[ERROR] Failed to capture frame from camera.")
            sys.exit(1)

    print(f"Image size: {color_image.shape[1]}x{color_image.shape[0]}")

    # --- Detect markers ---
    corners_list, ids_flat, centroids = detect_aruco_markers(color_image, aruco_dict)
    print(f"Detected {len(centroids)} ArUco markers: {list(centroids.keys())}")

    if len(centroids) == 0:
        print("[ERROR] No markers detected. Check lighting, focus, and marker print quality.")
        sys.exit(1)

    # --- Compute homography ---
    H, matched_img, matched_world, matched_ids = compute_homography(
        centroids, world_points_config
    )

    if H is None:
        print("[ERROR] Homography computation failed.")
        sys.exit(1)

    print(f"\nHomography matrix (image -> world):\n{H}")

    # --- Reprojection error ---
    errors, mean_error = compute_reprojection_error(H, matched_img, matched_world)
    print(f"\nReprojection error per marker:")
    for mid, img_pt, world_pt, err in zip(matched_ids, matched_img, matched_world, errors):
        print(f"  ID {mid}: image {img_pt} -> world {world_pt} | error: {err:.2f} mm")
    print(f"  Mean error: {mean_error:.2f} mm")

    # --- Plane depth ---
    plane_depth_mm = 0.0
    if depth_image is not None:
        kernel_size = config.get("detection", {}).get("depth_kernel_size", 5)
        plane_depth_mm = compute_plane_depth(depth_image, matched_img, kernel_size)
        print(f"\nAverage workspace plane depth: {plane_depth_mm:.1f} mm")

    # --- Save result ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        "homography_matrix": H.tolist(),
        "plane_depth_mm": plane_depth_mm,
        "aruco_dictionary": dict_name,
        "detected_markers": len(centroids),
        "matched_markers": len(matched_ids),
        "matched_marker_ids": matched_ids,
        "mean_reprojection_error_mm": mean_error,
        "per_marker_errors_mm": {str(mid): err for mid, err in zip(matched_ids, errors)},
        "timestamp": int(time.time()),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nCalibration saved to: {args.output}")

    # --- Save snapshot with marker world coordinates ---
    coord_image = color_image.copy()

    # Draw X and Y axes from the origin marker (ID 0) using the homography
    if "0" in centroids and H is not None:
        H_inv = np.linalg.inv(H)
        axis_len = 100.0  # mm in world space
        origin_world = np.array([[[0.0, 0.0]]], dtype=np.float32)
        x_end_world = np.array([[[axis_len, 0.0]]], dtype=np.float32)
        y_end_world = np.array([[[0.0, axis_len]]], dtype=np.float32)

        origin_px = cv2.perspectiveTransform(origin_world, H_inv).reshape(2).astype(int)
        x_end_px = cv2.perspectiveTransform(x_end_world, H_inv).reshape(2).astype(int)
        y_end_px = cv2.perspectiveTransform(y_end_world, H_inv).reshape(2).astype(int)

        # X axis (red)
        cv2.arrowedLine(coord_image, tuple(origin_px), tuple(x_end_px),
                        (0, 0, 255), 3, tipLength=0.15)
        cv2.putText(coord_image, "X", (x_end_px[0] + 10, x_end_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Y axis (green)
        cv2.arrowedLine(coord_image, tuple(origin_px), tuple(y_end_px),
                        (0, 255, 0), 3, tipLength=0.15)
        cv2.putText(coord_image, "Y", (y_end_px[0] + 10, y_end_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for tag_id_str, (cx, cy) in centroids.items():
        ix, iy = int(cx), int(cy)
        cv2.circle(coord_image, (ix, iy), 6, (0, 0, 255), -1)
        if tag_id_str in world_points_config:
            wx, wy = world_points_config[tag_id_str][:2]
            label = f"ID{tag_id_str} [{wx:.0f},{wy:.0f}]"
            cv2.putText(coord_image, label, (ix + 10, iy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    coord_path = os.path.join(os.path.dirname(args.output), "calibration_coords.png")
    cv2.imwrite(coord_path, coord_image)
    print(f"Coordinate snapshot saved to: {coord_path}")

    # --- Visualization ---
    if args.show:
        annotated = draw_calibration_result(
            color_image, corners_list, ids_flat, centroids,
            world_points_config, H, errors, matched_ids,
        )
        cv2.imshow("Calibration Result", annotated)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
