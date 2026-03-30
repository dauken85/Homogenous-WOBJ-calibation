# Homography Workspace Calibration

Automatic workspace calibration using ArUco markers and homography transforms. Detects ArUco markers with known world positions, computes a homography matrix (image → world), and uses it to map detected objects from pixel coordinates to real-world coordinates (mm).

Based on the approach described in Burde et al., IEEE CASE 2023.

## Overview

1. **Print** ArUco markers and place them at known positions in the workspace(Example PDF in repo)
2. **Calibrate** by capturing a frame and computing the homography
3. **Detect** objects and get their world coordinates (x, y, z)

## Project Structure

```
├── calibrate_workspace.py   # Main calibration script
├── camera.py                # Orbbec Gemini 2 camera wrapper (RGB-D)
├── detect_objects.py        # Object detection with world coordinate output
├── generate_markers.py      # Generate printable ArUco marker sheets
├── measure_markers.py       # Measure marker positions relative to origin
├── requirements.txt
├── config/
│   ├── workspace_config.json    # Marker layout and detection parameters
│   └── True_value.json          # Hand-measured ground truth positions
└── output/                      # Generated calibration results and images
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Requirements

- `opencv-contrib-python >= 4.8.0`
- `numpy >= 1.24.0`
- `pyorbbecsdk2` (for Orbbec Gemini 2 camera; optional if using image files)

## Configuration

Edit `config/workspace_config.json` to match your marker setup:

```json
{
    "aruco_dictionary": "DICT_5X5_100",
    "marker_size_mm": 50,
    "marker_ids": [0, 1, 2, 3, 4, 5],
    "world_points": {
        "0": [0.0, 0.0],
        "1": [145.0, 0.0],
        "2": [335.0, 0.0],
        "3": [0.0, 220.0],
        "4": [145.0, 220.0],
        "5": [335.0, 220.0]
    }
}
```

- `world_points`: `[x, y]` position in mm of each marker's center, relative to marker 0 as the origin.
- `marker_size_mm`: Physical side length of each printed marker.

## Usage

### 1. Generate Markers

```bash
python generate_markers.py
```

Produces printable marker images in `output/markers/`.

### 2. Calibrate Workspace

```bash
python calibrate_workspace.py           # Capture from camera
python calibrate_workspace.py --show    # Also display annotated result
python calibrate_workspace.py --image path/to/image.png  # Use saved image
```

Outputs:
- `output/calibration_result.json` — homography matrix and error metrics
- `output/calibration_coords.png` — snapshot with marker positions and X/Y axes
- `output/marker_axes.png` — snapshot with 3D axes drawn at each marker

### 3. Detect Objects

```bash
python detect_objects.py                # Live from camera
python detect_objects.py --image path/to/image.png
```

Uses the saved homography to transform detected object centroids to world coordinates (x, y in mm). Depth (z) is obtained from the aligned depth sensor as height above the calibrated workspace plane.

### 4. Measure Marker Positions (Optional)

```bash
python measure_markers.py
```

Uses `solvePnP` with the known marker size to estimate each marker's 3D position relative to marker 0. Compares against ground truth values in `config/True_value.json`.

## Camera

Uses the **Orbbec Gemini 2** RGB-D camera via `pyorbbecsdk2`. The `camera.py` module provides:

- `OrbbecCamera` — live capture with hardware-aligned RGB + depth
- `FileFallbackCamera` — loads image files for offline testing

If `pyorbbecsdk2` is not installed, the camera module falls back to file-based input.

## How It Works

1. ArUco markers are detected in the camera image
2. Detected marker centroids (pixels) are matched to their known world positions (mm)
3. A homography matrix **H** is computed: `world_point = H × image_point`
4. Reprojection error is computed to validate calibration accuracy
5. For object detection, the same **H** transforms any pixel location to world coordinates
