"""Region segmentation based on edge maps.

Uses connected-component labeling on edge-inverted images to extract
homogeneous intensity regions, as described in the Multi-GAN paper.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Region:
    """A segmented image region with metadata."""

    region_id: int
    mask: np.ndarray          # (H, W) boolean mask
    centroid: Tuple[float, float]  # (row, col)
    bbox: Tuple[int, int, int, int]  # (r1, c1, r2, c2) inclusive
    area: int
    smv: float = 0.0          # Segment Mean Value (computed later)
    cluster_id: int = -1      # Assigned cluster index


def segment_regions(
    edges: np.ndarray,
    min_region_size: int = 64,
) -> List[Region]:
    """Segment an image into regions by inverting edge map and using connected components.

    The edge map divides the image into enclosed regions. By inverting the edge
    map (edges become barriers) and applying morphological closing to bridge
    small gaps, we obtain contiguous regions via connected-component labeling.

    Args:
        edges:            Binary edge map (H, W), uint8, 255 = edge, 0 = no edge.
        min_region_size:  Minimum region area in pixels; smaller regions are discarded.

    Returns:
        List of Region dataclass instances.
    """
    # Invert edges: edges (255) become 0 (barrier), non-edges (0) become 255
    inv_edges = cv2.bitwise_not(edges)

    # Morphological close to bridge small gaps in edge barriers
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(inv_edges, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    regions: List[Region] = []
    # Skip label 0 (background)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_region_size:
            continue

        mask = (labels == lbl)
        cent = (float(centroids[lbl, 1]), float(centroids[lbl, 0]))  # (row, col)
        r1 = stats[lbl, cv2.CC_STAT_TOP]
        c1 = stats[lbl, cv2.CC_STAT_LEFT]
        r2 = r1 + stats[lbl, cv2.CC_STAT_HEIGHT] - 1
        c2 = c1 + stats[lbl, cv2.CC_STAT_WIDTH] - 1

        regions.append(Region(
            region_id=lbl,
            mask=mask,
            centroid=cent,
            bbox=(r1, c1, r2, c2),
            area=area,
        ))

    return regions


def compute_smv(
    gray: np.ndarray,
    regions: List[Region],
) -> None:
    """Compute Segment Mean Value (SMV) for each region and store it in-place.

    SMV is the mean grayscale intensity of pixels within the region,
    normalized to [0, 1].

    Args:
        gray:     Grayscale image (H, W), float in [0, 1] or uint8.
        regions:  List of Region objects (modified in-place).
    """
    if gray.dtype == np.uint8:
        gray_f = gray.astype(np.float32) / 255.0
    else:
        gray_f = gray.astype(np.float32)

    for region in regions:
        vals = gray_f[region.mask]
        region.smv = float(np.mean(vals)) if len(vals) > 0 else 0.0
