"""
Generate printable ArUco calibration markers.

Produces individual marker PNGs and a combined sheet for printing.
Uses the ArUco dictionary and marker IDs defined in workspace_config.json.
"""

import json
import os
import argparse

import cv2
import numpy as np


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

# Target DPI for printing
PRINT_DPI = 300


def mm_to_pixels(mm, dpi=PRINT_DPI):
    """Convert millimeters to pixels at the given DPI."""
    return int(mm / 25.4 * dpi)


def generate_single_marker(aruco_dict, marker_id, marker_size_px, border_bits=1):
    """
    Generate a single ArUco marker image with a white border.

    Args:
        aruco_dict: OpenCV ArUco dictionary object.
        marker_id: Integer ID of the marker.
        marker_size_px: Size of the marker in pixels (excluding border).
        border_bits: Width of the white border in marker-cell units.

    Returns:
        np.ndarray: Marker image (grayscale) with white border.
    """
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

    # Add white border
    cell_size = marker_size_px // (aruco_dict.markerSize + 2)
    border_px = cell_size * border_bits
    bordered = cv2.copyMakeBorder(
        marker_image,
        border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return bordered


def add_label(marker_image, marker_id, marker_size_mm, label_height_px=60):
    """
    Add a text label below the marker showing its ID and size.

    Args:
        marker_image: Grayscale marker image.
        marker_id: Integer ID.
        marker_size_mm: Physical size in mm.
        label_height_px: Height of the label area in pixels.

    Returns:
        np.ndarray: Marker image with label appended at the bottom.
    """
    w = marker_image.shape[1]
    label = np.full((label_height_px, w), 255, dtype=np.uint8)

    text = f"ID: {marker_id}  |  {marker_size_mm}mm  |  DICT_5X5"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (label_height_px + text_size[1]) // 2

    cv2.putText(label, text, (text_x, text_y), font, font_scale, 0, thickness)

    return np.vstack([marker_image, label])


def create_marker_sheet(marker_images, cols=3, padding_px=40):
    """
    Arrange multiple marker images into a single printable sheet.

    Args:
        marker_images: List of grayscale marker images (all same size).
        cols: Number of columns in the grid.
        padding_px: Padding between markers in pixels.

    Returns:
        np.ndarray: Combined sheet image.
    """
    if not marker_images:
        return np.full((100, 100), 255, dtype=np.uint8)

    h, w = marker_images[0].shape[:2]
    rows = (len(marker_images) + cols - 1) // cols

    sheet_w = cols * w + (cols + 1) * padding_px
    sheet_h = rows * h + (rows + 1) * padding_px
    sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

    for idx, img in enumerate(marker_images):
        r = idx // cols
        c = idx % cols
        y = padding_px + r * (h + padding_px)
        x = padding_px + c * (w + padding_px)
        sheet[y : y + h, x : x + w] = img

    return sheet


def main():
    parser = argparse.ArgumentParser(
        description="Generate printable ArUco calibration markers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "workspace_config.json"),
        help="Path to workspace config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "markers"),
        help="Output directory for marker images.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    dict_name = config["aruco_dictionary"]
    marker_size_mm = config["marker_size_mm"]
    marker_ids = config["marker_ids"]

    if dict_name not in ARUCO_DICT_MAP:
        print(f"[ERROR] Unknown ArUco dictionary: {dict_name}")
        print(f"  Available: {list(ARUCO_DICT_MAP.keys())}")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])
    marker_size_px = mm_to_pixels(marker_size_mm)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"ArUco dictionary : {dict_name}")
    print(f"Marker size      : {marker_size_mm} mm ({marker_size_px} px at {PRINT_DPI} DPI)")
    print(f"Marker IDs       : {marker_ids}")
    print(f"Output directory : {args.output_dir}")
    print()

    labeled_images = []

    for mid in marker_ids:
        marker_img = generate_single_marker(aruco_dict, mid, marker_size_px)
        labeled_img = add_label(marker_img, mid, marker_size_mm)
        labeled_images.append(labeled_img)

        # Save individual marker
        out_path = os.path.join(args.output_dir, f"marker_{mid}.png")
        cv2.imwrite(out_path, labeled_img)
        print(f"  Saved marker ID {mid} -> {out_path}")

    # Create and save combined sheet
    sheet = create_marker_sheet(labeled_images, cols=3)
    sheet_path = os.path.join(args.output_dir, "markers_sheet.png")
    cv2.imwrite(sheet_path, sheet)
    print(f"\n  Saved combined sheet -> {sheet_path}")

    print(f"\nDone. Print at {PRINT_DPI} DPI for correct physical size.")
    print("Tip: Measure printed markers with a ruler to verify they are "
          f"{marker_size_mm} mm on each side.")


if __name__ == "__main__":
    main()
