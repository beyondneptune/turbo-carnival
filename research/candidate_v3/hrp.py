"""Hierarchical Risk Parity implementation per López de Prado 2016.

Pure-numpy/scipy implementation (no third-party HRP libraries). The 3 steps:
1. Tree clustering on correlation-distance matrix
2. Quasi-diagonalization (reorder so similar assets are adjacent)
3. Recursive bisection: at each node, allocate inversely proportional to
   subcluster variance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt((1 - ρ_ij) / 2). Clamped to [0, 1]."""
    d = np.sqrt(np.clip((1 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def cluster_assets(corr: np.ndarray, linkage_method: str = "single") -> np.ndarray:
    """Hierarchical clustering link matrix."""
    dist = correlation_to_distance(corr)
    # scipy linkage wants the upper triangle as a condensed distance vector
    condensed = ssd.squareform(dist, checks=False)
    return sch.linkage(condensed, method=linkage_method)


def quasi_diagonal(link: np.ndarray) -> list[int]:
    """Sort assets so that similar ones are adjacent — López de Prado 2016 §5."""
    link = link.astype(int)
    n = link.shape[0] + 1
    # The "root" cluster is the last merge: link[-1, 0..1] are its children
    cur = list(link[-1, :2])

    def expand(node: int) -> list[int]:
        if node < n:
            return [node]
        merge_row = link[node - n]
        left, right = int(merge_row[0]), int(merge_row[1])
        return expand(left) + expand(right)

    return expand(cur[0]) + expand(cur[1])


def get_cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
    """Inverse-variance weighted variance of a cluster."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    iv_w = 1.0 / np.diag(cov_slice)
    iv_w = iv_w / iv_w.sum()
    return float(iv_w @ cov_slice @ iv_w)


def hrp_weights(returns: pd.DataFrame, linkage_method: str = "single",
                weight_floor: float = 0.0, weight_cap: float = 1.0) -> pd.Series:
    """Compute HRP weights from a DataFrame of returns (rows=days, cols=assets).

    Returns a pd.Series indexed by asset symbol, weights sum to 1.
    """
    cov = returns.cov().values
    corr = returns.corr().values
    link = cluster_assets(corr, linkage_method=linkage_method)
    sort_idx = quasi_diagonal(link)
    n = len(sort_idx)

    # Recursive bisection
    weights = np.ones(n)
    clusters = [sort_idx]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left = c[:half]
            right = c[half:]
            v_l = get_cluster_var(cov, left)
            v_r = get_cluster_var(cov, right)
            # Allocate inversely proportional to subcluster variance
            alpha = 1.0 - v_l / (v_l + v_r) if (v_l + v_r) > 1e-12 else 0.5
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= 1.0 - alpha
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters

    # Apply floor + cap
    if weight_floor > 0:
        weights = np.maximum(weights, weight_floor)
    if weight_cap < 1.0:
        weights = np.minimum(weights, weight_cap)
    # Renormalise to sum to 1
    weights = weights / weights.sum()

    return pd.Series(weights, index=returns.columns)


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """The 1/N benchmark."""
    n = len(returns.columns)
    return pd.Series(np.ones(n) / n, index=returns.columns)
