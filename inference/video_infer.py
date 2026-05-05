"""Video colorization with temporal harmony.

Extends image colorization to video by tracking regions across frames
and applying a harmony blending mechanism to maintain temporal consistency
and reduce flickering.
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from preprocessing.edge_detection import get_edge_map
from preprocessing.segmentation import segment_regions, compute_smv, Region
from preprocessing.clustering import cluster_regions, compute_dynamic_k
from models.multi_gan import MultiGAN
from inference.image_infer import ImageColorizer


@dataclass
class TrackedRegion:
    """A region tracked across video frames."""

    region: Region
    frame_idx: int
    keypoint: cv2.KeyPoint
    descriptor: np.ndarray
    color_history: List[np.ndarray]  # list of (3,) mean color vectors


class VideoColorizer:
    """Video colorization engine with temporal harmony.

    Uses SIFT (or ORB as fallback) feature matching to track regions
    between consecutive frames, then blends colorized outputs to
    reduce flickering artifacts.
    """

    def __init__(
        self,
        multi_gan: MultiGAN,
        config: dict,
        device: torch.device = torch.device("cpu"),
    ):
        self.multi_gan = multi_gan
        self.device = device

        pre_cfg = config.get("preprocessing", {})
        self.edge_method = pre_cfg.get("edge_method", "canny")
        self.canny_low = pre_cfg.get("canny_low", 50)
        self.canny_high = pre_cfg.get("canny_high", 150)
        self.min_region_size = pre_cfg.get("min_region_size", 64)

        clust_cfg = config.get("clustering", {})
        self.gamma = clust_cfg.get("gamma", 0.05)
        self.k_min = clust_cfg.get("k_min", 10)
        self.k_max = clust_cfg.get("k_max", 64)

        inf_cfg = config.get("inference", {})
        self.alpha = inf_cfg.get("alpha", 0.5)  # harmony blend factor
        self.feature_detector_type = inf_cfg.get("feature_detector", "sift")
        self.match_threshold = inf_cfg.get("match_threshold", 0.7)

        # Initialize feature detector
        self._init_feature_detector()

        # Image colorizer for per-frame processing
        self.image_colorizer = ImageColorizer(multi_gan, config, device)

        # Previous frame state for tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_regions: List[Region] = []
        self.prev_colorized: Optional[np.ndarray] = None
        self.tracked_regions: Dict[int, TrackedRegion] = {}

    def _init_feature_detector(self) -> None:
        """Initialize SIFT or fall back to ORB."""
        if self.feature_detector_type == "sift":
            try:
                self.detector = cv2.SIFT_create()
                self.norm_type = cv2.NORM_L2
            except AttributeError:
                print("[WARNING] SIFT not available (patent issue). Falling back to ORB.")
                self.detector = cv2.ORB_create(nfeatures=500)
                self.norm_type = cv2.NORM_HAMMING
        else:
            self.detector = cv2.ORB_create(nfeatures=500)
            self.norm_type = cv2.NORM_HAMMING

        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

    def _extract_region_features(
        self, gray: np.ndarray, regions: List[Region]
    ) -> List[Tuple[cv2.KeyPoint, np.ndarray]]:
        """Extract a representative keypoint + descriptor per region.

        For each region, computes its centroid as a keypoint and runs
        the descriptor on a small patch around the centroid.

        Args:
            gray:    Grayscale frame (H, W), uint8.
            regions: List of segmented Region objects.

        Returns:
            List of (keypoint, descriptor) for each region.
        """
        features = []
        for region in regions:
            r, c = region.centroid
            kp = cv2.KeyPoint(float(c), float(r), size=float(region.area) ** 0.5)

            # Extract descriptor from centroid patch
            patch_size = 32
            r_start = max(0, int(r) - patch_size // 2)
            r_end = min(gray.shape[0], r_start + patch_size)
            c_start = max(0, int(c) - patch_size // 2)
            c_end = min(gray.shape[1], c_start + patch_size)

            patch = gray[r_start:r_end, c_start:c_end]
            if patch.size == 0:
                features.append((kp, np.zeros((128,), dtype=np.float32)))
                continue

            # Detect keypoints in patch and compute descriptors
            kps, descs = self.detector.detectAndCompute(patch, None)
            if descs is not None and len(descs) > 0:
                features.append((kp, descs[0]))
            else:
                # Fallback: zero descriptor
                desc_size = 128 if self.feature_detector_type == "sift" else 32
                features.append((kp, np.zeros((desc_size,), dtype=np.float32)))

        return features

    def _match_regions(
        self,
        prev_features: List[Tuple[cv2.KeyPoint, np.ndarray]],
        curr_features: List[Tuple[cv2.KeyPoint, np.ndarray]],
        prev_regions: List[Region],
        curr_regions: List[Region],
    ) -> Dict[int, int]:
        """Match regions between consecutive frames.

        Uses Lowe's ratio test for robust matching.

        Returns:
            Dictionary mapping current_region_index -> previous_region_index.
        """
        matches_map: Dict[int, int] = {}

        if len(prev_features) == 0 or len(curr_features) == 0:
            return matches_map

        # Build descriptor arrays
        prev_desc = np.array([f[1] for f in prev_features], dtype=np.float32)
        curr_desc = np.array([f[1] for f in curr_features], dtype=np.float32)

        if prev_desc.ndim != 2 or curr_desc.ndim != 2:
            return matches_map
        if prev_desc.shape[1] != curr_desc.shape[1]:
            return matches_map

        # Match using BFMatcher with ratio test
        try:
            raw_matches = self.matcher.knnMatch(curr_desc, prev_desc, k=2)

            for i, match_pair in enumerate(raw_matches):
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    curr_idx = m.queryIdx
                    prev_idx = m.trainIdx
                    if curr_idx < len(curr_regions) and prev_idx < len(prev_regions):
                        matches_map[curr_idx] = prev_idx
        except cv2.error:
            pass

        return matches_map

    def _apply_harmony(
        self,
        current_color: np.ndarray,  # (H, W, 3) uint8 per-region color
        region_mask: np.ndarray,     # (H, W) boolean
        prev_colorized: np.ndarray,  # (H, W, 3) uint8
    ) -> np.ndarray:
        """Apply temporal harmony blending for a matched region.

        harmony_color = alpha * current + (1 - alpha) * previous

        Args:
            current_color:  Current frame's colorized output for this region.
            region_mask:    Boolean mask of the region.
            prev_colorized: Previous frame's full colorized output.
            alpha:          Blend weight for current frame.

        Returns:
            Blended color array (H, W, 3), uint8.
        """
        alpha = self.alpha

        # Blend only within the region mask
        blended = current_color.copy().astype(np.float32)
        mask_3ch = np.stack([region_mask] * 3, axis=-1)

        prev_region = prev_colorized.astype(np.float32)
        blended[mask_3ch] = (
            alpha * current_color.astype(np.float32)[mask_3ch] +
            (1.0 - alpha) * prev_region[mask_3ch]
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def colorize_frame(
        self, gray: np.ndarray, frame_idx: int = 0
    ) -> np.ndarray:
        """Colorize a single video frame with temporal harmony.

        Args:
            gray:      (H, W) grayscale frame, uint8.
            frame_idx: Frame index (0 for first frame).

        Returns:
            (H, W, 3) colorized RGB frame, uint8.
        """
        H, W = gray.shape
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)

        # --- Base colorization (full image) ---
        colorized = self.image_colorizer.colorize(gray)

        if frame_idx == 0 or self.prev_colorized is None:
            # First frame: no harmony blending
            self.prev_gray = gray.copy()
            self.prev_colorized = colorized.copy()

            # Store regions for next frame matching
            edges = get_edge_map(gray, method=self.edge_method)
            self.prev_regions = segment_regions(edges, min_region_size=self.min_region_size)
            gray_float = gray.astype(np.float32) / 255.0
            compute_smv(gray_float, self.prev_regions)

            # Cluster and assign
            if len(self.prev_regions) > 0:
                smv_vals = np.array([r.smv for r in self.prev_regions], dtype=np.float32)
                k = compute_dynamic_k(gray, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max)
                labels, _ = cluster_regions(smv_vals, k=k, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max)
                for i, r in enumerate(self.prev_regions):
                    r.cluster_id = int(labels[i])

            return colorized

        # --- Subsequent frames: apply harmony ---
        # Segment current frame
        edges = get_edge_map(gray, method=self.edge_method)
        curr_regions = segment_regions(edges, min_region_size=self.min_region_size)

        if len(curr_regions) == 0 or len(self.prev_regions) == 0:
            self.prev_gray = gray.copy()
            self.prev_colorized = colorized.copy()
            self.prev_regions = curr_regions
            return colorized

        gray_float = gray.astype(np.float32) / 255.0
        compute_smv(gray_float, curr_regions)

        if len(curr_regions) > 0:
            smv_vals = np.array([r.smv for r in curr_regions], dtype=np.float32)
            k = compute_dynamic_k(gray, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max)
            labels, _ = cluster_regions(smv_vals, k=k, gamma=self.gamma, k_min=self.k_min, k_max=self.k_max)
            for i, r in enumerate(curr_regions):
                r.cluster_id = int(labels[i])

        # Extract features and match regions
        prev_feats = self._extract_region_features(self.prev_gray, self.prev_regions)
        curr_feats = self._extract_region_features(gray, curr_regions)
        matches = self._match_regions(prev_feats, curr_feats, self.prev_regions, curr_regions)

        # Apply harmony to matched regions
        harmonized = colorized.copy()
        for curr_idx, prev_idx in matches.items():
            curr_region = curr_regions[curr_idx]
            prev_region = self.prev_regions[prev_idx]

            # Blend colors within the current region's mask
            harmonized = self._apply_harmony(
                harmonized,
                curr_region.mask,
                self.prev_colorized,
            )

        # Update state
        self.prev_gray = gray.copy()
        self.prev_colorized = harmonized.copy()
        self.prev_regions = curr_regions

        return harmonized

    def reset(self) -> None:
        """Reset temporal state (call at video boundaries)."""
        self.prev_gray = None
        self.prev_regions = []
        self.prev_colorized = None
        self.tracked_regions = {}


def colorize_video(
    multi_gan: MultiGAN,
    input_path: str,
    output_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
    fps: Optional[float] = None,
) -> None:
    """Colorize a full video file.

    Args:
        multi_gan:   Trained MultiGAN ensemble.
        input_path:  Path to input grayscale video.
        output_path: Path for output colorized video.
        config:      Configuration dict.
        device:      Torch device.
        fps:         Output FPS (auto-detected from input if None).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colorizer = VideoColorizer(multi_gan, config, device)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale if input is color
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Colorize with harmony
        colorized = colorizer.colorize_frame(gray, frame_idx)

        # Write output (BGR for OpenCV)
        colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
        out.write(colorized_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed frame {frame_idx}")

    cap.release()
    out.release()
    print(f"Video colorized: {output_path} ({frame_idx} frames)")
