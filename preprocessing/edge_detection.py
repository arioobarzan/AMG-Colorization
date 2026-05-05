"""Edge detection using OpenCV Canny, approximating the Dollar edge method.

The original paper references the Dollar edge detector. We use Canny as an
efficient, accessible approximation that produces structurally similar edge
maps for region segmentation.
"""

import cv2
import numpy as np
from typing import Tuple


def detect_edges_canny(
    gray: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
) -> np.ndarray:
    """Detect edges in a grayscale image using the Canny algorithm.

    Args:
        gray:           Grayscale image (H, W), uint8, range [0, 255].
        low_threshold:  Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.
        aperture_size:  Sobel kernel size.

    Returns:
        Binary edge map (H, W), uint8.
    """
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    # Gaussian blur to reduce noise, then Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    return edges


def detect_edges_dollar(
    gray: np.ndarray,
) -> np.ndarray:
    """Structured-edge approximation of the Dollar detector.

    Uses gradient magnitude thresholding on multiple scales, which follows
    the spirit of the Dollar edge box approach.

    Args:
        gray: Grayscale image (H, W), uint8 or float.

    Returns:
        Binary edge map (H, W), uint8.
    """
    if gray.dtype != np.uint8:
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = gray.copy()

    # Multi-scale Sobel gradients
    edges_sum = np.zeros_like(gray_uint8, dtype=np.float32)
    scales = [(3, 1.0), (5, 1.5)]

    for ksize, sigma in scales:
        blurred = cv2.GaussianBlur(gray_uint8, (ksize, ksize), sigma)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        edges_sum += mag

    # Normalize and threshold
    edges_sum = edges_sum / len(scales)
    edges_norm = (edges_sum / edges_sum.max() * 255).astype(np.uint8)
    _, edges_bin = cv2.threshold(edges_norm, 30, 255, cv2.THRESH_BINARY)
    return edges_bin


def get_edge_map(
    gray: np.ndarray,
    method: str = "canny",
    **kwargs,
) -> np.ndarray:
    """Unified edge detection interface.

    Args:
        gray:   Grayscale image (H, W).
        method: "canny" or "dollar".
        **kwargs: Passed to the specific detector.

    Returns:
        Binary edge map (H, W), uint8.
    """
    if method == "canny":
        return detect_edges_canny(
            gray,
            low_threshold=kwargs.get("low_threshold", 50),
            high_threshold=kwargs.get("high_threshold", 150),
        )
    elif method == "dollar":
        return detect_edges_dollar(gray)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
