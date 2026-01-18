# backend/soil_sampling_engine.py
from __future__ import annotations

import hashlib
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

ENV_VARS = ["slope", "BSI", "CEC", "mrvbf", "NDVI", "total_clay", "twi"]


# -------------------------
# Utils
# -------------------------
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


def _fit_transformer(X_full: np.ndarray, mode: str):
    """Fit a transformation on the *full* set and return a callable.

    Why: for representativeness metrics (BD/KLD) you need both the full
    candidate set and the sample set in the *same* transformed space.
    """

    X_full = np.asarray(X_full, dtype=np.float64)
    if mode == "none":
        return lambda X: np.asarray(X, dtype=np.float64)

    if mode == "zscore":
        mu = X_full.mean(axis=0)
        sd = X_full.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        return lambda X: (np.asarray(X, dtype=np.float64) - mu) / sd

    if mode in {"rank_uniform", "rank_normal"}:
        # Empirical CDF fitted on full, applied to sample via searchsorted.
        # This keeps both sets on a consistent scale.
        cols_sorted = [np.sort(X_full[:, j]) for j in range(X_full.shape[1])]
        n = float(len(X_full))

        def _to_u(col_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(col_sorted, x, side="left").astype(np.float64)
            u = (idx + 0.5) / n
            return np.clip(u, 1e-6, 1.0 - 1e-6)

        if mode == "rank_uniform":

            def _tx(X):
                X = np.asarray(X, dtype=np.float64)
                out = np.empty_like(X, dtype=np.float64)
                for j in range(X.shape[1]):
                    out[:, j] = _to_u(cols_sorted[j], X[:, j])
                return out

            return _tx

        # rank_normal
        from scipy.stats import norm

        def _tx(X):
            X = np.asarray(X, dtype=np.float64)
            out = np.empty_like(X, dtype=np.float64)
            for j in range(X.shape[1]):
                out[:, j] = norm.ppf(_to_u(cols_sorted[j], X[:, j]))
            return out

        return _tx

    raise ValueError("scale_mode must be one of: none, zscore, rank_uniform, rank_normal")


def _extract_indices_from_clhs(res, n_samples: int) -> np.ndarray:
    if isinstance(res, dict):
        for k in ("sample_indices", "indices", "idx"):
            if k in res:
                return np.asarray(res[k], dtype=int)
        for v in res.values():
            a = np.asarray(v)
            if a.ndim == 1 and len(a) == n_samples:
                return a.astype(int)
        raise ValueError(f"Unexpected clhs dict output keys={list(res.keys())}")

    a = np.asarray(res)
    if a.ndim == 1:
        return a.astype(int)
    if a.ndim == 2 and a.shape[0] == n_samples and a.shape[1] == 1:
        return a[:, 0].astype(int)

    raise ValueError(f"Unexpected clhs output shape={a.shape}")


# -------------------------
# Core sampler (your CLHS)
# -------------------------
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
    Returns subset selected by CLHS.
    Never raises due to CLHS: falls back to stable random sample.
    """
    df = df_candidates.copy()

    covariates = covariates or ENV_VARS
    covariates = [c for c in covariates if c in df.columns]

    cols = list(covariates) + (["x", "y"] if include_xy_in_clhs else [])
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).reset_index(drop=True)

    N = len(df)
    if N == 0:
        return df
    n_samples = int(min(max(1, n_samples), N))
    if n_samples >= N:
        return df.copy()

    seed = _stable_seed(polygon_geojson, n_samples=n_samples, base_seed=base_seed)
    rng = np.random.default_rng(seed)

    if len(cols) < 2 or n_samples <= 1:
        idx = rng.choice(N, size=n_samples, replace=False)
        return df.iloc[idx].copy()

    X = df[cols].to_numpy(dtype=np.float64, copy=True)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Xs = _transform_matrix(X, mode=scale_mode)

    if n_iterations is None:
        n_iterations = int(min(30000, max(3000, 120 * n_samples)))

    try:
        from clhs import clhs
        res = clhs(Xs, n_samples, seed=int(seed), n_iterations=int(n_iterations), progress=False)
        idx = _extract_indices_from_clhs(res, n_samples=n_samples)
        idx = np.asarray(idx, dtype=int)

        if idx.ndim != 1 or len(idx) != n_samples or idx.min() < 0 or idx.max() >= N:
            raise ValueError("CLHS returned invalid indices")

        return df.iloc[idx].copy()

    except Exception:
        idx = rng.choice(N, size=n_samples, replace=False)
        return df.iloc[idx].copy()


# -------------------------
# Recommendation logic
# -------------------------
def _cov_mle(X: np.ndarray) -> np.ndarray:
    """Maximum-likelihood covariance (divide by N, not N-1)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = X.shape[0]
    if n <= 1:
        return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    Xm = X - X.mean(axis=0, keepdims=True)
    return (Xm.T @ Xm) / float(n)


def _regularize_cov(S: np.ndarray, reg_covar: float) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    d = S.shape[0]
    return S + float(reg_covar) * np.eye(d, dtype=np.float64)


def _gaussian_kl(m0: np.ndarray, S0: np.ndarray, m1: np.ndarray, S1: np.ndarray) -> float:
    """KL divergence D_KL(N0 || N1) for Gaussians."""
    m0 = np.asarray(m0, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    S0 = np.asarray(S0, dtype=np.float64)
    S1 = np.asarray(S1, dtype=np.float64)
    d = int(m0.size)

    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    if sign0 <= 0 or sign1 <= 0:
        # should not happen after regularization, but keep it safe
        return float("inf")

    # tr(S1^{-1} S0)
    tr_term = float(np.trace(np.linalg.solve(S1, S0)))

    # (m1 - m0)^T S1^{-1} (m1 - m0)
    delta = (m1 - m0).reshape(-1, 1)
    quad = float((delta.T @ np.linalg.solve(S1, delta)).squeeze())

    return float(0.5 * (tr_term + quad - d + (logdet1 - logdet0)))


def _gaussian_bhattacharyya(m0: np.ndarray, S0: np.ndarray, m1: np.ndarray, S1: np.ndarray) -> float:
    """Bhattacharyya distance between Gaussians."""
    m0 = np.asarray(m0, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    S0 = np.asarray(S0, dtype=np.float64)
    S1 = np.asarray(S1, dtype=np.float64)
    d = int(m0.size)

    S = 0.5 * (S0 + S1)
    signS, logdetS = np.linalg.slogdet(S)
    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    if signS <= 0 or sign0 <= 0 or sign1 <= 0:
        return float("inf")

    delta = (m1 - m0).reshape(-1, 1)
    quad = float((delta.T @ np.linalg.solve(S, delta)).squeeze())
    term1 = 0.125 * quad
    term2 = 0.5 * (logdetS - 0.5 * (logdet0 + logdet1))
    return float(term1 + term2)


def _quantile_match_score(full: np.ndarray, sample: np.ndarray, qs: np.ndarray) -> float:
    """
    Compare sample quantiles to full quantiles, normalized by IQR.
    Lower is better.
    """
    full = full[np.isfinite(full)]
    sample = sample[np.isfinite(sample)]
    if full.size < 5 or sample.size < 2:
        return 0.0

    q_full = np.quantile(full, qs)
    q_s = np.quantile(sample, qs)

    iqr = np.quantile(full, 0.75) - np.quantile(full, 0.25)
    denom = float(iqr) if iqr > 1e-12 else float(np.std(full)) if np.std(full) > 1e-12 else 1.0

    return float(np.mean(np.abs(q_s - q_full) / denom))


def representativeness_score(
    df_full: pd.DataFrame,
    df_sample: pd.DataFrame,
    covariates: List[str],
    scale_mode: str = "rank_normal",
    metric: str = "bd",
    reg_covar: float = 1e-6,
) -> float:
    """
    Representativeness score. Lower = sample matches full better.

    metric:
      - "bd": Bhattacharyya distance between multivariate Gaussians
      - "kld": symmetrized KL divergence between multivariate Gaussians
      - "quantile": legacy per-covariate quantile matching
    """

    metric = (metric or "bd").strip().lower()
    covariates = [c for c in covariates if c in df_full.columns and c in df_sample.columns]
    if not covariates:
        return 0.0

    # numeric + drop rows with NaNs in any covariate
    Xf = df_full[covariates].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    Xs = df_sample[covariates].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    Xf = Xf[np.isfinite(Xf).all(axis=1)]
    Xs = Xs[np.isfinite(Xs).all(axis=1)]

    if Xf.shape[0] < 5 or Xs.shape[0] < 2:
        return 0.0

    if metric == "quantile":
        qs = np.arange(0.05, 1.0, 0.05)
        scores = []
        for j, c in enumerate(covariates):
            s = _quantile_match_score(Xf[:, j], Xs[:, j], qs)
            scores.append(s)
        return float(np.mean(scores)) if scores else 0.0

    # Fit transform on full; apply consistently to sample
    tx = _fit_transformer(Xf, mode=scale_mode)
    Xf_t = tx(Xf)
    Xs_t = tx(Xs)

    m0 = Xf_t.mean(axis=0)
    m1 = Xs_t.mean(axis=0)
    S0 = _regularize_cov(_cov_mle(Xf_t), reg_covar=reg_covar)
    S1 = _regularize_cov(_cov_mle(Xs_t), reg_covar=reg_covar)

    if metric in {"bd", "bha", "bhattacharyya"}:
        return _gaussian_bhattacharyya(m0, S0, m1, S1)

    if metric in {"kld", "kl", "klsym", "symkl"}:
        return 0.5 * (_gaussian_kl(m0, S0, m1, S1) + _gaussian_kl(m1, S1, m0, S0))

    raise ValueError("metric must be one of: bd, kld, quantile")


def _default_n_grid(N: int) -> List[int]:
    """
    A small grid (fast for web) that still gives a nice curve.
    """
    if N <= 12:
        return list(range(1, N + 1))
    # Denser at the low end so tier suggestions don't collapse to the same N
    # just because the coarse grid skipped over the true threshold-crossing.
    base = list(range(3, 21)) + [25, 30, 40, 50, 60]
    grid = sorted({n for n in base if n <= N})
    # Always include the upper anchor for the curve (helps stabilize gains)
    top = min(60, N)
    if top not in grid:
        grid.append(top)
    return grid


def recommend_sample_sizes(
    df_candidates: pd.DataFrame,
    polygon_geojson: Optional[dict],
    covariates: Optional[List[str]] = None,
    cost_per_sample: float = 25.0,
    include_xy_in_clhs: bool = False,
    scale_mode: str = "rank_normal",
    rep_metric: str = "bd",
    reg_covar: float = 1e-6,
    n_grid: Optional[List[int]] = None,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Returns:
      - candidates_count
      - curve: [{n, score, gain, cost}, ...]
      - tiers: suggested n for weak/medium/good/perfect
    """
    covariates = covariates or ENV_VARS
    covariates = [c for c in covariates if c in df_candidates.columns]

    df = df_candidates.copy()
    N = len(df)
    if N == 0:
        return {
            "candidates_count": 0,
            "curve": [],
            "tiers": {"weak": 0, "medium": 0, "good": 0, "perfect": 0},
        }

    n_grid = n_grid or _default_n_grid(N)
    n_grid = [int(n) for n in n_grid if 1 <= int(n) <= N]
    n_grid = sorted(set(n_grid))
    if not n_grid:
        n_grid = [min(30, N)]

    # compute curve (distance-based score: lower is better)
    curve = []
    scores = []
    for n in n_grid:
        df_s = suggest_clhs_samples(
            df,
            n_samples=n,
            polygon_geojson=polygon_geojson,
            covariates=covariates,
            include_xy_in_clhs=include_xy_in_clhs,
            scale_mode=scale_mode,
            base_seed=base_seed,
        )
        sc = representativeness_score(
            df,
            df_s,
            covariates=covariates,
            scale_mode=scale_mode,
            metric=rep_metric,
            reg_covar=reg_covar,
        )
        scores.append(sc)
        curve.append({"n": int(n), "score": float(sc), "cost": float(n * cost_per_sample)})

    # smooth to avoid jitter (monotone non-increasing)
    scores = np.asarray(scores, dtype=float)
    sm = np.minimum.accumulate(scores)  # running min
    for i, v in enumerate(sm):
        curve[i]["score"] = float(v)

    # Convert distance score -> coverage (0..1, higher is better)
    # For Gaussians, Bhattacharyya coefficient is exp(-BD) which is a true overlap measure.
    # For sym-KL, we use a bounded heuristic exp(-0.5 * symKL).
    rep_m = (rep_metric or "bd").strip().lower()

    def _score_to_coverage(s: float) -> float:
        if not np.isfinite(s):
            return 0.0
        if rep_m in {"bd", "bha", "bhattacharyya"}:
            return float(np.exp(-float(s)))
        if rep_m in {"kld", "kl", "klsym", "symkl"}:
            return float(np.exp(-0.5 * float(s)))
        # fallback
        return float(np.exp(-float(s)))

    cov = np.asarray([_score_to_coverage(v) for v in sm], dtype=float)
    # monotone non-decreasing (coverage should increase with n)
    cov = np.maximum.accumulate(cov)
    for i, v in enumerate(cov):
        curve[i]["coverage"] = float(np.clip(v, 0.0, 1.0))

    # gains (0..1) as relative improvement in coverage across the evaluated grid
    c0 = float(cov[0])
    cL = float(cov[-1])
    denom = (cL - c0) if abs(cL - c0) > 1e-12 else 1.0
    for i, _ in enumerate(curve):
        gain = (float(cov[i]) - c0) / denom
        curve[i]["gain"] = float(np.clip(gain, 0.0, 1.0))

    # tiers from *coverage* thresholds (matches the UI wording)
    def tier_n(thr: float) -> int:
        for row in curve:
            if float(row.get("coverage", 0.0)) >= thr:
                return int(row["n"])
        return int(curve[-1]["n"])

    tiers = {
        "weak": tier_n(0.60),
        "medium": tier_n(0.75),
        "good": tier_n(0.88),
        "perfect": tier_n(0.95),
    }

    return {
        "candidates_count": int(N),
        "curve": curve,
        "tiers": tiers,
    }


# -------------------------
# Replicates
# -------------------------
def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    R = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def add_replicates(
    df_candidates: pd.DataFrame,
    df_selected: pd.DataFrame,
    replicate_fraction: float,
    replicate_radius_m: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds replicate points chosen from candidates near selected points.
    Returns (df_primary, df_all_selected_with_reps)
    """
    replicate_fraction = float(np.clip(replicate_fraction, 0.0, 0.5))
    replicate_radius_m = float(max(0.0, replicate_radius_m))

    if replicate_fraction <= 0.0 or replicate_radius_m <= 0.0:
        df_primary = df_selected.copy()
        df_all = df_selected.copy()
        df_all["is_replicate"] = False
        df_all["replicate_of"] = None
        return df_primary, df_all

    rng = np.random.default_rng(int(seed))

    df_candidates = df_candidates.copy().reset_index(drop=True)
    df_selected = df_selected.copy().reset_index(drop=True)

    df_selected_ids = set(df_selected.index.tolist())
    df_selected["is_replicate"] = False
    df_selected["replicate_of"] = None

    n_total = len(df_selected)
    n_rep = int(round(n_total * replicate_fraction))
    if n_rep <= 0:
        return df_selected, df_selected

    used_candidate_idx = set()
    # try to identify unique id column, fallback to (lon,lat)
    def key_row(r):
        if "id" in r and pd.notna(r["id"]):
            return ("id", int(r["id"]))
        return ("xy", float(r["lon"]), float(r["lat"]))

    selected_keys = set(key_row(r) for _, r in df_selected.iterrows())

    reps = []
    for _ in range(n_rep):
        # pick a primary point at random
        j = int(rng.integers(0, n_total))
        lon0 = float(df_selected.loc[j, "lon"])
        lat0 = float(df_selected.loc[j, "lat"])

        # find nearest candidate within radius not already selected
        best_idx = None
        best_d = None
        for i, r in df_candidates.iterrows():
            if i in used_candidate_idx:
                continue
            k = key_row(r)
            if k in selected_keys:
                continue
            d = _haversine_m(lon0, lat0, float(r["lon"]), float(r["lat"]))
            if d <= replicate_radius_m:
                if best_d is None or d < best_d:
                    best_d = d
                    best_idx = i

        if best_idx is None:
            continue

        used_candidate_idx.add(best_idx)
        rr = df_candidates.loc[best_idx].copy()
        rr["is_replicate"] = True
        rr["replicate_of"] = int(df_selected.loc[j, "id"]) if "id" in df_selected.columns and pd.notna(df_selected.loc[j, "id"]) else j
        reps.append(rr)

    if not reps:
        return df_selected, df_selected

    df_rep = pd.DataFrame(reps).reset_index(drop=True)
    df_all = pd.concat([df_selected, df_rep], ignore_index=True)
    return df_selected, df_all
