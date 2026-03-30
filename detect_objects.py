"""
Detect objects and return their position in world coordinates (x, y, z).

Uses a saved homography calibration to transform detected object centroids
from image pixels to world coordinates (mm). Z is obtained from the
aligned depth sensor, reported as height above the calibrated workspace plane.
"""

import json
import os
import sys
import argparse

import cv2
import numpy as np

from camera import get_depth_at_pixel


def load_calibration(calibration_path):
    """
    Load a saved calibration result.

    Args:
        calibration_path: Path to calibration_result.json.

    Returns:
        tuple: (homography_matrix, plane_depth_mm)
    """
    with open(calibration_path, "r") as f:
        data = json.load(f)

    H = np.array(data["homography_matrix"], dtype=np.float64)
    plane_depth_mm = data.get("plane_depth_mm", 0.0)
    return H, plane_depth_mm


def detect_objects_by_color(color_image, hsv_lower, hsv_upper,
                            min_area=500, max_area=100000):
    """
    Detect objects using HSV color thresholding and contour analysis.

    Args:
        color_image: BGR image.
        hsv_lower: Lower HSV bound as [H, S, V].
        hsv_upper: Upper HSV bound as [H, S, V].
        min_area: Minimum contour area in pixels.
        max_area: Maximum contour area in pixels.

    Returns:
        list of dict: Detected objects with keys:
            - pixel_x, pixel_y: Centroid in image coordinates.
            - contour: OpenCV contour array.
            - area_px: Contour area in pixels.
            - bbox: (x, y, w, h) bounding rectangle.
    """
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        bbox = cv2.boundingRect(cnt)

        detections.append({
            "pixel_x": cx,
            "pixel_y": cy,
            "contour": cnt,
            "area_px": area,
            "bbox": bbox,
        })

    return detections


def transform_to_world(detections, H, depth_image=None,
                       plane_depth_mm=0.0, depth_kernel_size=5):
    """
    Transform detected object positions from image to world coordinates.

    Args:
        detections: List of detection dicts (from detect_objects_by_color).
        H: 3x3 homography matrix (image -> world).
        depth_image: Optional depth image (H, W) in mm. If None, z=0.
        plane_depth_mm: Average depth of the workspace plane in mm.
        depth_kernel_size: Kernel size for depth averaging at each point.

    Returns:
        list of dict: Each with keys:
            - x_mm, y_mm: World coordinates in mm.
            - z_mm: Height above workspace plane in mm (0 if no depth).
            - depth_mm: Raw depth reading in mm.
            - pixel_x, pixel_y: Image centroid.
            - area_px: Contour area.
            - bbox: Bounding rectangle (x, y, w, h).
    """
    results = []

    for det in detections:
        px, py = det["pixel_x"], det["pixel_y"]

        # Transform pixel -> world via homography
        img_pt = np.array([[[px, py]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(img_pt, H)[0][0]
        wx, wy = float(world_pt[0]), float(world_pt[1])

        # Get depth and compute height above plane
        depth_mm = 0.0
        z_mm = 0.0
        if depth_image is not None:
            ix = int(round(px))
            iy = int(round(py))
            depth_mm = get_depth_at_pixel(depth_image, ix, iy, depth_kernel_size)
            if depth_mm > 0 and plane_depth_mm > 0:
                z_mm = plane_depth_mm - depth_mm

        results.append({
            "x_mm": round(wx, 2),
            "y_mm": round(wy, 2),
            "z_mm": round(z_mm, 2),
            "depth_mm": round(depth_mm, 2),
            "pixel_x": round(px, 1),
            "pixel_y": round(py, 1),
            "area_px": round(det["area_px"], 1),
            "bbox": det["bbox"],
        })

    return results


def draw_detections(color_image, results):
    """
    Draw bounding boxes and world coordinate labels on the image.

    Args:
        color_image: BGR image (will be copied).
        results: List of result dicts from transform_to_world.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = color_image.copy()

    for i, r in enumerate(results):
        x, y, w, h = r["bbox"]
        cx, cy = int(r["pixel_x"]), int(r["pixel_y"])

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Centroid
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        # World coordinate label
        label = f"({r['x_mm']:.0f}, {r['y_mm']:.0f}, {r['z_mm']:.0f})mm"
        cv2.putText(annotated, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Object index
        cv2.putText(annotated, f"#{i}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Detect objects and return their world coordinates (x, y, z)."
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "calibration_result.json"),
        help="Path to calibration result JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "workspace_config.json"),
        help="Path to workspace config JSON (for detection settings).",
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
        "--hsv-lower",
        type=str,
        default=None,
        help="Override HSV lower bound as 'H,S,V' (e.g. '35,50,50').",
    )
    parser.add_argument(
        "--hsv-upper",
        type=str,
        default=None,
        help="Override HSV upper bound as 'H,S,V' (e.g. '85,255,255').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated detection image.",
    )
    args = parser.parse_args()

    # --- Load calibration ---
    print(f"Loading calibration from: {args.calibration}")
    H, plane_depth_mm = load_calibration(args.calibration)
    print(f"Plane depth: {plane_depth_mm:.1f} mm")

    # --- Load detection config ---
    with open(args.config, "r") as f:
        config = json.load(f)

    det_config = config.get("detection", {})
    hsv_lower = det_config.get("hsv_lower", [35, 50, 50])
    hsv_upper = det_config.get("hsv_upper", [85, 255, 255])
    min_area = det_config.get("min_contour_area", 500)
    max_area = det_config.get("max_contour_area", 100000)
    depth_kernel = det_config.get("depth_kernel_size", 5)

    # CLI overrides
    if args.hsv_lower:
        hsv_lower = [int(v) for v in args.hsv_lower.split(",")]
    if args.hsv_upper:
        hsv_upper = [int(v) for v in args.hsv_upper.split(",")]

    print(f"HSV range: {hsv_lower} -> {hsv_upper}")
    print(f"Area range: {min_area} -> {max_area} px")

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

    # --- Detect objects ---
    detections = detect_objects_by_color(
        color_image, hsv_lower, hsv_upper, min_area, max_area
    )
    print(f"\nDetected {len(detections)} object(s)")

    # --- Transform to world coordinates ---
    results = transform_to_world(
        detections, H, depth_image, plane_depth_mm, depth_kernel
    )

    # --- Print results ---
    print(f"\n{'#':<4} {'X (mm)':>8} {'Y (mm)':>8} {'Z (mm)':>8} {'Depth':>8} {'Pixel':>14} {'Area':>8}")
    print("-" * 62)
    for i, r in enumerate(results):
        print(f"{i:<4} {r['x_mm']:>8.1f} {r['y_mm']:>8.1f} {r['z_mm']:>8.1f} "
              f"{r['depth_mm']:>8.1f} ({r['pixel_x']:>5.0f},{r['pixel_y']:>5.0f}) "
              f"{r['area_px']:>8.0f}")

    # --- Visualization ---
    if args.show:
        annotated = draw_detections(color_image, results)
        cv2.imshow("Object Detection", annotated)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


if __name__ == "__main__":
    main()
