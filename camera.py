"""
Orbbec Gemini 2 camera wrapper.

Provides aligned RGB + depth capture via pyorbbecsdk.
Falls back to loading image files if pyorbbecsdk is not available.
"""

import numpy as np
import cv2

try:
    from pyorbbecsdk import (
        Pipeline,
        Config,
        OBSensorType,
        OBStreamType,
        OBAlignMode,
        OBFormat,
    )

    ORBBEC_AVAILABLE = True
except ImportError:
    ORBBEC_AVAILABLE = False
    print("[WARNING] pyorbbecsdk not found. Camera will use file-based fallback.")


class OrbbecCamera:
    """Wrapper for Orbbec Gemini 2 RGB-D camera."""

    def __init__(self):
        if not ORBBEC_AVAILABLE:
            raise RuntimeError(
                "pyorbbecsdk is not installed. "
                "Install it or use OrbbecCamera.from_files() instead."
            )
        self.pipeline = Pipeline()
        config = Config()

        # Enable color stream
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # Enable depth stream — request Y16 (raw uint16) to avoid compressed formats
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profiles.get_video_stream_profile(
                0, 0, OBFormat.Y16, 0
            )
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # Enable hardware alignment (depth aligned to color)
        config.set_align_mode(OBAlignMode.HW_MODE)

        self.pipeline.start(config)

    def capture(self):
        """
        Capture one aligned RGB + depth frame.

        Returns:
            tuple: (color_image, depth_image)
                - color_image: np.ndarray BGR (H, W, 3) uint8
                - depth_image: np.ndarray depth in mm (H, W) uint16
                Returns (None, None) if capture fails.
        """
        frameset = self.pipeline.wait_for_frames(1000)
        if frameset is None:
            return None, None

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        # Convert color frame to numpy BGR
        color_data = np.asanyarray(color_frame.get_data())
        color_format = color_frame.get_format()

        if color_format == OBFormat.RGB:
            color_image = cv2.cvtColor(color_data.reshape(
                (color_frame.get_height(), color_frame.get_width(), 3)
            ), cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.YUYV:
            color_data = color_data.reshape(
                (color_frame.get_height(), color_frame.get_width(), 2)
            )
            color_image = cv2.cvtColor(color_data, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        else:
            # Assume BGR
            color_image = color_data.reshape(
                (color_frame.get_height(), color_frame.get_width(), 3)
            )

        # Convert depth frame to numpy (uint16, values in mm)
        depth_h = depth_frame.get_height()
        depth_w = depth_frame.get_width()
        expected_pixels = depth_h * depth_w
        depth_data = np.asanyarray(depth_frame.get_data())

        if depth_data.size == expected_pixels * 2:
            # Raw uint8 buffer containing uint16 values
            depth_image = depth_data.view(np.uint16).reshape((depth_h, depth_w))
        elif depth_data.size == expected_pixels:
            # Already one element per pixel (likely uint16 view)
            depth_image = depth_data.reshape((depth_h, depth_w)).astype(np.uint16)
        else:
            print(f"[WARNING] Unexpected depth data size {depth_data.size} "
                  f"for {depth_w}x{depth_h} frame. Returning zero depth.")
            depth_image = np.zeros((depth_h, depth_w), dtype=np.uint16)

        # Apply depth scale if needed
        depth_scale = depth_frame.get_depth_scale()
        if depth_scale != 1.0:
            depth_image = (depth_image.astype(np.float32) * depth_scale).astype(np.uint16)

        return color_image, depth_image

    def release(self):
        """Stop the pipeline and release resources."""
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class FileFallbackCamera:
    """Fallback camera that loads color and depth images from files."""

    def __init__(self, color_path, depth_path=None):
        """
        Args:
            color_path: Path to BGR image file (PNG, JPG, etc.)
            depth_path: Path to depth image. Supports:
                        - .npy file (uint16, values in mm)
                        - .png file (16-bit single channel, values in mm)
                        If None, depth will be a zero array.
        """
        self.color_image = cv2.imread(color_path)
        if self.color_image is None:
            raise FileNotFoundError(f"Could not load color image: {color_path}")

        if depth_path is not None:
            if depth_path.endswith(".npy"):
                self.depth_image = np.load(depth_path).astype(np.uint16)
            else:
                self.depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if self.depth_image is None:
                    raise FileNotFoundError(f"Could not load depth image: {depth_path}")
                self.depth_image = self.depth_image.astype(np.uint16)
        else:
            h, w = self.color_image.shape[:2]
            self.depth_image = np.zeros((h, w), dtype=np.uint16)

    def capture(self):
        """Return the loaded images."""
        return self.color_image.copy(), self.depth_image.copy()

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def get_depth_at_pixel(depth_image, x, y, kernel_size=5):
    """
    Get averaged depth value at a pixel location.

    Uses a kernel_size x kernel_size window centered on (x, y),
    averaging only non-zero values to handle invalid depth pixels.

    Args:
        depth_image: np.ndarray (H, W) uint16, depth in mm.
        x: Pixel column coordinate.
        y: Pixel row coordinate.
        kernel_size: Size of the averaging window (odd number).

    Returns:
        float: Averaged depth in mm, or 0.0 if no valid depth.
    """
    h, w = depth_image.shape[:2]
    half = kernel_size // 2

    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)

    patch = depth_image[y0:y1, x0:x1].astype(np.float64)
    valid = patch[patch > 0]

    if len(valid) == 0:
        return 0.0

    return float(np.mean(valid))
