# backend/soil_sampling_engine.py
from __future__ import annotations

import hashlib
from typing import Optional, List

import numpy as np
import pandas as pd

ENV_VARS = ["slope", "BSI", "CEC", "mrvbf", "NDVI", "total_clay", "twi"]


def _stable_seed(polygon_geojson: Optional[dict], n_samples: int, base_seed: int = 42) -> int:
    if polygon_geojson is None:
        return int(base_seed)
    s = f"{base_seed}|{n_samples}|{polygon_geojson}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _rank_uniform(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    u = (r - 0.5) / len(x)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def _transform_matrix(X: np.ndarray, mode: str) -> np.ndarray:
    # X is (N, D), float64
    if mode == "none":
        return X

    if mode == "zscore":
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        return (X - mu) / sd

    if mode == "rank_uniform":
        cols = [_rank_uniform(X[:, j]) for j in range(X.shape[1])]
        return np.vstack(cols).T

    if mode == "rank_normal":
        from scipy.stats import norm
        cols = [norm.ppf(_rank_uniform(X[:, j])) for j in range(X.shape[1])]
        return np.vstack(cols).T

    raise ValueError("scale_mode must be one of: none, zscore, rank_uniform, rank_normal")


def _extract_indices_from_clhs(res, n_samples: int) -> np.ndarray:
    """
    clhs() return types can vary by version. Make extraction robust.
    """
    if isinstance(res, dict):
        for k in ("sample_indices", "indices", "idx"):
            if k in res:
                return np.asarray(res[k], dtype=int)
        # last resort: find a 1D array-like of length n_samples
        for v in res.values():
            a = np.asarray(v)
            if a.ndim == 1 and len(a) == n_samples:
                return a.astype(int)
        raise ValueError(f"Unexpected clhs dict output keys={list(res.keys())}")

    a = np.asarray(res)
    if a.ndim == 1:
        return a.astype(int)

    # Some versions may return shape (n_samples, 1) etc.
    if a.ndim == 2 and a.shape[0] == n_samples and a.shape[1] == 1:
        return a[:, 0].astype(int)

    raise ValueError(f"Unexpected clhs output shape={a.shape}")


def suggest_clhs_samples(
    df_candidates: pd.DataFrame,
    n_samples: int,
    polygon_geojson: Optional[dict] = None,
    covariates: Optional[List[str]] = None,
    include_xy_in_clhs: bool = False,
    scale_mode: str = "rank_normal",
    base_seed: int = 42,
    n_iterations: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a subset of df_candidates selected by CLHS.
    Never raises due to CLHS: it will fallback to a stable random sample.
    """

    df = df_candidates.copy()

    covariates = covariates or ENV_VARS
    covariates = [c for c in covariates if c in df.columns]

    # If there are too few covariates, CLHS can be unstable; fallback
    cols = list(covariates) + (["x", "y"] if include_xy_in_clhs else [])

    # Force numeric scalars
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing required CLHS columns
    df = df.dropna(subset=cols).reset_index(drop=True)

    N = len(df)
    if N == 0:
        return df
    n_samples = int(min(max(1, n_samples), N))
    if n_samples >= N:
        return df.copy()

    seed = _stable_seed(polygon_geojson, n_samples=n_samples, base_seed=base_seed)
    rng = np.random.default_rng(seed)

    # If only 0/1 dimensions or n_samples very small, skip CLHS
    if len(cols) < 2 or n_samples <= 1:
        idx = rng.choice(N, size=n_samples, replace=False)
        return df.iloc[idx].copy()

    X = df[cols].to_numpy(dtype=np.float64, copy=True)
    X = np.ascontiguousarray(X, dtype=np.float64)

    Xs = _transform_matrix(X, mode=scale_mode)

    if n_iterations is None:
        # keeps it responsive
        n_iterations = int(min(30000, max(3000, 120 * n_samples)))

    try:
        from clhs import clhs
        res = clhs(Xs, n_samples, seed=int(seed), n_iterations=int(n_iterations), progress=False)
        idx = _extract_indices_from_clhs(res, n_samples=n_samples)
        idx = np.asarray(idx, dtype=int)

        # Guard against any weird indices
        if idx.ndim != 1 or len(idx) != n_samples or idx.min() < 0 or idx.max() >= N:
            raise ValueError("CLHS returned invalid indices")

        return df.iloc[idx].copy()

    except Exception:
        # Stable fallback: never break the webapp
        idx = rng.choice(N, size=n_samples, replace=False)
        return df.iloc[idx].copy()
