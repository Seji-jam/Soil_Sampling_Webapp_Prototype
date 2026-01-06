import hashlib
import pandas as pd
import numpy as np
from scipy.stats import norm
from clhs import clhs

ENV_VARS = ['slope', 'BSI', 'CEC', 'mrvbf', 'NDVI', 'total_clay', 'twi']

# ---- transforms (copied/simplified from your analysis script) ----
def _rank_uniform_1d(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return (r - 0.5) / len(x)

def _rank_normal_1d(x: np.ndarray) -> np.ndarray:
    u = _rank_uniform_1d(x)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u)

def transform_matrix_from_full(X_full: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return X_full.astype(float, copy=False)
    if mode == "zscore":
        mu = X_full.mean(0)
        sd = X_full.std(0, ddof=0)
        sd[sd == 0] = 1.0
        return (X_full - mu) / sd
    if mode == "rank_uniform":
        cols = [_rank_uniform_1d(X_full[:, j]) for j in range(X_full.shape[1])]
        return np.vstack(cols).T
    if mode == "rank_normal":
        cols = [_rank_normal_1d(X_full[:, j]) for j in range(X_full.shape[1])]
        return np.vstack(cols).T
    raise ValueError("mode must be one of {'zscore','rank_uniform','rank_normal','none'}")

def transform_df_columns_from_full(df: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    X = df[cols].to_numpy(float)
    X_t = transform_matrix_from_full(X, mode=mode)
    out = pd.DataFrame(X_t, columns=cols, index=df.index)
    return out

# ---- basic cleaning (keep it light for webapp) ----
def clean_covariates(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    # ensure numeric + drop missing
    tmp = df.copy()
    for c in covariates + ["x", "y"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # optional: handle negative clay like your analysis script
    if "total_clay" in tmp.columns:
        tmp.loc[tmp["total_clay"] < 0, "total_clay"] = np.nan

    # optional: winsorize CEC extremes (same logic as your script)
    if "CEC" in tmp.columns:
        valid = tmp["CEC"].dropna()
        if len(valid) > 0:
            qlo, qhi = valid.quantile([0.005, 0.995]).to_numpy()
            tmp["CEC"] = tmp["CEC"].clip(lower=qlo, upper=qhi)

    tmp = tmp.dropna(subset=covariates + ["x", "y"]).reset_index(drop=True)
    return tmp

def _clhs_indices(cov_df_scaled: pd.DataFrame, n_total: int, seed: int, n_iterations: int) -> np.ndarray:
    res = clhs(cov_df_scaled, n_total, progress=False, n_iterations=n_iterations, seed=seed)

    # robust extraction (because clhs sometimes returns dicts)
    if isinstance(res, dict):
        if "sample_indices" in res: return np.asarray(res["sample_indices"], dtype=int)
        if "indices" in res:        return np.asarray(res["indices"], dtype=int)
        for v in res.values():
            if isinstance(v, (list, np.ndarray)) and len(v) == n_total:
                return np.asarray(v, dtype=int)
        raise ValueError(f"clhs result missing indices: keys={list(res.keys())}")

    return np.asarray(res, dtype=int)

def _stable_seed_from_request(polygon_geojson: dict, n_samples: int, base_seed: int = 42) -> int:
    s = (str(n_samples) + "|" + str(base_seed) + "|" +
         hashlib.sha256(str(polygon_geojson).encode("utf-8")).hexdigest())
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
    iters_per_point: int = 120,
    iters_min: int = 3000,
    iters_max: int = 30000,
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

    cov_vars = list(covariates) + (["x", "y"] if include_xy_in_clhs else [])
    cov_scaled = transform_df_columns_from_full(df, cov_vars, mode=scale_mode)

    if n_iterations is None:
        n_iterations = int(max(iters_min, min(iters_max, iters_per_point * n_samples)))

    seed = base_seed
    if polygon_geojson is not None:
        seed = _stable_seed_from_request(polygon_geojson, n_samples=n_samples, base_seed=base_seed)

    idx = _clhs_indices(cov_scaled, n_total=n_samples, seed=seed, n_iterations=n_iterations)
    return df.iloc[idx].copy()