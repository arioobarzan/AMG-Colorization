"""Dynamic K-Means clustering based on Segment Mean Value (SMV).

The core idea from the Multi-GAN paper:
1. Compute grayscale histogram.
2. Select intensities whose frequency >= gamma.
3. Compute k = |V| / division_factor.
4. Clamp k to [k_min, k_max].
5. Cluster the SMV values of each region into k groups using K-Means.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional


def compute_dynamic_k(
    gray: np.ndarray,
    gamma: float = 0.05,
    division_factor: int = 4,
    k_min: int = 10,
    k_max: int = 64,
    bins: int = 256,
) -> int:
    """Compute the dynamic number of clusters (k) from the grayscale histogram.

    Args:
        gray:            Grayscale image (H, W), float in [0, 1] or uint8.
        gamma:           Minimum frequency threshold for an intensity to be counted.
        division_factor: k = |V| / division_factor.
        k_min:           Lower clamp bound.
        k_max:           Upper clamp bound.
        bins:            Number of histogram bins.

    Returns:
        Integer k: number of clusters.
    """
    if gray.dtype != np.uint8:
        gray_img = (gray * 255).astype(np.uint8)
    else:
        gray_img = gray

    h = gray_img.size  # total pixels

    # Compute histogram
    hist, _ = np.histogram(gray_img, bins=bins, range=(0, 255))
    # P(i) = hist[i] / h
    prob = hist / h

    # Select intensities with P(i) >= gamma
    significant = np.where(prob >= gamma)[0]

    k_raw = len(significant) // division_factor
    k = int(np.clip(k_raw, k_min, k_max))
    return max(k, 2)  # at least 2 clusters


def cluster_regions(
    smv_values: np.ndarray,
    k: Optional[int] = None,
    gray: Optional[np.ndarray] = None,
    gamma: float = 0.05,
    k_min: int = 10,
    k_max: int = 64,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """Cluster regions based on their SMV values using K-Means.

    Args:
        smv_values:  (N,) float array of SMV per region.
        k:           Number of clusters. If None, computed dynamically from `gray`.
        gray:        Grayscale image (required if k is None).
        gamma:       Frequency threshold for dynamic k.
        k_min:       Minimum k.
        k_max:       Maximum k.
        random_state: Seed for reproducibility.

    Returns:
        labels:  (N,) integer cluster labels.
        kmeans:  Fitted KMeans model.
    """
    if k is None:
        if gray is None:
            raise ValueError("Either `k` or `gray` must be provided.")
        k = compute_dynamic_k(gray, gamma=gamma, k_min=k_min, k_max=k_max)

    # Reshape for sklearn
    X = smv_values.reshape(-1, 1)

    # Ensure k does not exceed number of samples
    k_actual = min(k, len(X))

    kmeans = KMeans(n_clusters=k_actual, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans
