import hashlib
import numpy as np
import pandas as pd

ENV_VARS = ['slope', 'BSI', 'CEC', 'mrvbf', 'NDVI', 'total_clay', 'twi']

def clean_covariates(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    for c in covariates + ["x", "y"]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    keep = [c for c in covariates if c in tmp.columns]
    tmp = tmp.dropna(subset=keep + ["x", "y"]).reset_index(drop=True)
    return tmp

def _rank_uniform_1d(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return (r - 0.5) / len(x)

def _rank_normal_1d(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    u = _rank_uniform_1d(x)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u)

def transform_df(df: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    X = df[cols].to_numpy(float)
    if mode == "zscore":
        mu = X.mean(0)
        sd = X.std(0, ddof=0)
        sd[sd == 0] = 1.0
        X2 = (X - mu) / sd
    elif mode == "rank_normal":
        X2 = np.vstack([_rank_normal_1d(X[:, j]) for j in range(X.shape[1])]).T
    elif mode == "rank_uniform":
        X2 = np.vstack([_rank_uniform_1d(X[:, j]) for j in range(X.shape[1])]).T
    elif mode == "none":
        X2 = X
    else:
        raise ValueError("scale_mode must be one of: zscore, rank_normal, rank_uniform, none")
    return pd.DataFrame(X2, columns=cols, index=df.index)

def _stable_seed(polygon_geojson: dict | None, n_samples: int, base_seed: int) -> int:
    if polygon_geojson is None:
        return base_seed
    s = f"{base_seed}|{n_samples}|{polygon_geojson}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)

def suggest_clhs_samples(
    df_candidates: pd.DataFrame,
    n_samples: int,
    polygon_geojson: dict | None = None,
    covariates: list[str] | None = None,
    include_xy_in_clhs: bool = False,
    scale_mode: str = "rank_normal",
    base_seed: int = 42,
    n_iterations: int | None = None,
) -> pd.DataFrame:

    covariates = covariates or ENV_VARS
    covariates = [c for c in covariates if c in df_candidates.columns]

    df = clean_covariates(df_candidates, covariates=covariates)
    N = len(df)
    if N == 0:
        return df

    n_samples = int(min(max(1, n_samples), N))
    if n_samples >= N:
        return df.copy()

    cols = list(covariates) + (["x", "y"] if include_xy_in_clhs else [])
    scaled = transform_df(df, cols, mode=scale_mode)

    if n_iterations is None:
        n_iterations = int(min(30000, max(3000, 120 * n_samples)))

    seed = _stable_seed(polygon_geojson, n_samples=n_samples, base_seed=base_seed)

    try:
        from clhs import clhs
    except Exception as e:
        raise RuntimeError("Missing dependency 'clhs'. Add 'clhs' (and 'scipy') to requirements.txt") from e

    res = clhs(scaled, n_samples, seed=seed, n_iterations=n_iterations, progress=False)
    idx = np.asarray(res, dtype=int)
    return df.iloc[idx].copy()
