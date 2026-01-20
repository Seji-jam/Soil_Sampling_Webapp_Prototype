from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

try:
    # Repo layout: backend/main.py and backend/soil_sampling_engine.py
    from backend.soil_sampling_engine import (
        ENV_VARS,
        recommend_sample_sizes,
        suggest_clhs_samples,
    )
except Exception:
    # Local/dev layout: main.py next to soil_sampling_engine.py
    from soil_sampling_engine import (
        ENV_VARS,
        recommend_sample_sizes,
        suggest_clhs_samples,
    )


# -----------------------------------------------------------------------------
# Config: COG URLs (public https://storage.googleapis.com/... or signed URLs)
#
# Preferred: set env var COVARIATE_URLS_JSON to a JSON object like:
# {
#   "ref": "https://storage.googleapis.com/<bucket>/dem_slope/dem_3dep_10m_18157.tif",
#   "slope": "https://storage.googleapis.com/<bucket>/dem_slope/slope_10m_18157.tif",
#   "BSI": "https://storage.googleapis.com/<bucket>/bsi/bsi_july_5yr_18157.tif",
#   "CEC": "https://storage.googleapis.com/<bucket>/SSURGO/cec_0_30cm_cog.tif",
#   "mrvbf": "https://storage.googleapis.com/<bucket>/dem_slope/mrvbf_18157.tif",
#   "NDVI": "https://storage.googleapis.com/<bucket>/ndvi_july_5yr/tiles/ndvi_july_5yr_18157.tif",
#   "total_clay": "https://storage.googleapis.com/<bucket>/SSURGO/total_clay_0_30cm_cog.tif",
#   "twi": "https://storage.googleapis.com/<bucket>/dem_slope/twi_18157.tif"
# }
#
# ("ref" is used for extent/window geometry; it can be DEM or any aligned raster.)
# -----------------------------------------------------------------------------


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"


def _normalize_gcs_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if u.startswith("gs://"):
        # gs://bucket/path -> https://storage.googleapis.com/bucket/path
        rest = u[len("gs://") :]
        parts = rest.split("/", 1)
        if len(parts) == 1:
            return f"https://storage.googleapis.com/{parts[0]}"
        return f"https://storage.googleapis.com/{parts[0]}/{parts[1]}"
    return u


def _load_covariate_urls() -> Dict[str, str]:
    # 1) JSON in env
    if os.getenv("COVARIATE_URLS_JSON"):
        urls = json.loads(os.environ["COVARIATE_URLS_JSON"])
        return {k: _normalize_gcs_url(v) for k, v in urls.items()}

    # 2) JSON file path in env
    if os.getenv("COVARIATE_URLS_PATH"):
        p = Path(os.environ["COVARIATE_URLS_PATH"]).expanduser()
        urls = json.loads(p.read_text())
        return {k: _normalize_gcs_url(v) for k, v in urls.items()}

    # 3) optional repo file (if you create it)
    p = ROOT_DIR / "data" / "covariate_urls.json"
    if p.exists():
        urls = json.loads(p.read_text())
        return {k: _normalize_gcs_url(v) for k, v in urls.items()}

    raise RuntimeError(
        "Missing COG manifest. Set env var COVARIATE_URLS_JSON (recommended) or COVARIATE_URLS_PATH."
    )


COG_URLS = _load_covariate_urls()
REF_URL = COG_URLS.get("ref") or COG_URLS.get("slope") or COG_URLS.get("NDVI")
if not REF_URL:
    raise RuntimeError("COG manifest must include at least 'ref' (or 'slope'/'NDVI') URL.")


def _make_valid_polygon(g):
    try:
        from shapely.validation import make_valid

        return make_valid(g)
    except Exception:
        return g.buffer(0)


# -----------------------------------------------------------------------------
# Reference raster metadata (cached)
# -----------------------------------------------------------------------------


def _read_ref_meta() -> Tuple[Tuple[float, float, float, float], str]:
    """Return (bounds, crs_wkt_or_epsg_string) for the reference raster."""
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with rasterio.open(REF_URL) as ds:
            b = ds.bounds  # left, bottom, right, top in raster CRS
            crs = ds.crs
            return (float(b.left), float(b.bottom), float(b.right), float(b.top)), crs.to_string()


_REF_BOUNDS_5070, _REF_CRS = _read_ref_meta()
_TO_LL = Transformer.from_crs(_REF_CRS, "EPSG:4326", always_xy=True).transform
_TO_RASTER = Transformer.from_crs("EPSG:4326", _REF_CRS, always_xy=True).transform

# Extent in lon/lat for map
_minx, _miny, _maxx, _maxy = _REF_BOUNDS_5070
_corners_ll = [
    _TO_LL(_minx, _miny),
    _TO_LL(_minx, _maxy),
    _TO_LL(_maxx, _miny),
    _TO_LL(_maxx, _maxy),
]
MIN_LON = float(min(c[0] for c in _corners_ll))
MAX_LON = float(max(c[0] for c in _corners_ll))
MIN_LAT = float(min(c[1] for c in _corners_ll))
MAX_LAT = float(max(c[1] for c in _corners_ll))


app = FastAPI()


class SamplingRequest(BaseModel):
    polygon: Dict[str, Any]  # GeoJSON geometry
    n_samples: int = 30


class RecommendRequest(BaseModel):
    polygon: Dict[str, Any]
    cost_per_sample: float = 25.0
    metric: str = "bd"  # "bd" (Bhattacharyya) or "kld" (sym KL)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/extent")
def extent():
    return {"bounds": [[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]]}


# -----------------------------------------------------------------------------
# Candidate extraction from raster stack (adaptive)
# -----------------------------------------------------------------------------


def _adaptive_stride_m(area_m2: float, target_candidates: int = 50_000) -> int:
    """Pick a grid spacing (meters) to keep candidate count manageable."""
    if not np.isfinite(area_m2) or area_m2 <= 0:
        return 10
    # expected candidates approx area / stride^2
    stride = int(round(np.sqrt(area_m2 / float(max(1, target_candidates)))))
    # snap UP to 10 m multiples (since rasters are 10 m)
    stride = int(max(10, 10 * int(np.ceil(stride / 10))))
    return int(min(stride, 200))  # hard cap to avoid extremely sparse grids


_CAND_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_CAND_TTL_S = 600


def _cand_cache_key(poly: dict) -> str:
    return hashlib.sha256(str(poly).encode("utf-8")).hexdigest()


def _raster_candidates(poly_geojson: Dict[str, Any]) -> pd.DataFrame:
    """Create a candidate dataframe by sampling the aligned raster stack inside polygon."""

    key = _cand_cache_key(poly_geojson)
    now = time.time()
    if key in _CAND_CACHE:
        t0, df = _CAND_CACHE[key]
        if now - t0 < _CAND_TTL_S:
            return df.copy()

    poly_ll = _make_valid_polygon(shape(poly_geojson))
    if poly_ll.is_empty:
        return pd.DataFrame(columns=["x", "y", "lon", "lat"] + ENV_VARS)

    # quick bbox reject in lon/lat
    minx, miny, maxx, maxy = poly_ll.bounds
    if (maxx < MIN_LON) or (minx > MAX_LON) or (maxy < MIN_LAT) or (miny > MAX_LAT):
        return pd.DataFrame(columns=["x", "y", "lon", "lat"] + ENV_VARS)

    # project polygon to raster CRS (EPSG:5070)
    poly_5070 = _make_valid_polygon(shp_transform(_TO_RASTER, poly_ll))
    if poly_5070.is_empty:
        return pd.DataFrame(columns=["x", "y", "lon", "lat"] + ENV_VARS)

    area_m2 = float(poly_5070.area)
    stride_m = _adaptive_stride_m(area_m2)
    stride_px = max(1, int(round(stride_m / 10.0)))  # 10 m rasters

    # clip bounds to ref raster bounds
    bx0, by0, bx1, by1 = poly_5070.bounds
    rx0, ry0, rx1, ry1 = _REF_BOUNDS_5070
    bx0, by0 = max(bx0, rx0), max(by0, ry0)
    bx1, by1 = min(bx1, rx1), min(by1, ry1)
    if bx1 <= bx0 or by1 <= by0:
        return pd.DataFrame(columns=["x", "y", "lon", "lat"] + ENV_VARS)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with rasterio.open(REF_URL) as ref:
            window = from_bounds(bx0, by0, bx1, by1, transform=ref.transform)
            window = window.round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                return pd.DataFrame(columns=["x", "y", "lon", "lat"] + ENV_VARS)

            # Downsample on-the-fly to keep IO and memory bounded.
            # Each output pixel represents ~stride_px original pixels.
            win_tx = ref.window_transform(window)

            out_h = int(np.ceil(float(window.height) / float(stride_px)))
            out_w = int(np.ceil(float(window.width) / float(stride_px)))
            out_h = max(1, out_h)
            out_w = max(1, out_w)

            # Scale transform so the downsampled grid lines up with the window.
            sx = float(window.width) / float(out_w)
            sy = float(window.height) / float(out_h)
            ds_tx = win_tx * Affine.scale(sx, sy)

            inside = geometry_mask(
                [mapping(poly_5070)],
                out_shape=(out_h, out_w),
                transform=ds_tx,
                invert=True,
                all_touched=False,
            )

            # Read covariate rasters at the same downsampled shape
            cols: dict[str, np.ndarray] = {}
            ok = inside.copy()
            for name in ENV_VARS:
                url = COG_URLS.get(name)
                if not url:
                    raise RuntimeError(f"Missing URL for covariate '{name}' in manifest")
                with rasterio.open(url) as ds:
                    arr = ds.read(
                        1,
                        window=window,
                        out_shape=(out_h, out_w),
                        resampling=Resampling.nearest,
                    ).astype(np.float32, copy=False)
                    nodata = ds.nodata
                    if nodata is not None:
                        arr[arr == np.float32(nodata)] = np.nan
                    cols[name] = arr
                    ok &= np.isfinite(arr)

            if not np.any(ok):
                return pd.DataFrame(columns=["x", "y", "lon", "lat"] + ENV_VARS)

            rr, cc = np.where(ok)

            # Extract 1-D covariate vectors for candidate points
            col_vals: dict[str, np.ndarray] = {name: cols[name][rr, cc] for name in ENV_VARS}

            # Coordinates of downsampled pixel centers (vectorized)
            xs = ds_tx.c + (cc.astype(np.float64) + 0.5) * ds_tx.a + (rr.astype(np.float64) + 0.5) * ds_tx.b
            ys = ds_tx.f + (cc.astype(np.float64) + 0.5) * ds_tx.d + (rr.astype(np.float64) + 0.5) * ds_tx.e

    # convert to lon/lat
    lon, lat = Transformer.from_crs(_REF_CRS, "EPSG:4326", always_xy=True).transform(xs, ys)

    df = pd.DataFrame({"x": xs, "y": ys, "lon": lon, "lat": lat})
    for name in ENV_VARS:
        df[name] = col_vals[name]

    # Stable id
    df.insert(0, "id", np.arange(1, len(df) + 1, dtype=np.int64))

    # cache
    _CAND_CACHE[key] = (now, df)
    return df.copy()


# -----------------------------------------------------------------------------
# small cache for /api/recommend-n (keeps Render snappy)
# -----------------------------------------------------------------------------


_RECO_CACHE: dict[str, tuple[float, dict]] = {}
_RECO_TTL_S = 600


def _cache_key(poly: dict, cost_per_sample: float, metric: str) -> str:
    metric = (metric or "bd").strip().lower()
    return hashlib.sha256((str(poly) + f"|{cost_per_sample}|{metric}").encode("utf-8")).hexdigest()


@app.post("/api/recommend-n")
def recommend_n(req: RecommendRequest):
    try:
        metric = (req.metric or "bd").strip().lower()
        key = _cache_key(req.polygon, float(req.cost_per_sample), metric)
        now = time.time()

        if key in _RECO_CACHE:
            t0, payload = _RECO_CACHE[key]
            if now - t0 < _RECO_TTL_S:
                return payload

        df_sub = _raster_candidates(req.polygon)
        if df_sub.empty:
            payload = {
                "candidates_count": 0,
                "curve": [],
                "tiers": {"weak": 0, "medium": 0, "good": 0, "perfect": 0},
            }
            _RECO_CACHE[key] = (now, payload)
            return payload

        payload = recommend_sample_sizes(
            df_candidates=df_sub,
            polygon_geojson=req.polygon,
            cost_per_sample=float(req.cost_per_sample),
            include_xy_in_clhs=False,
            scale_mode="rank_normal",
            rep_metric=metric,
        )
        _RECO_CACHE[key] = (now, payload)
        return payload

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/suggest-samples")
def suggest_samples(req: SamplingRequest):
    try:
        df_sub = _raster_candidates(req.polygon)
        if df_sub.empty:
            return {"type": "FeatureCollection", "features": []}

        n_samples = int(min(max(1, req.n_samples), len(df_sub)))

        df_sel = suggest_clhs_samples(
            df_sub,
            n_samples=n_samples,
            polygon_geojson=req.polygon,
            include_xy_in_clhs=False,
            scale_mode="rank_normal",
        )

        features = []
        for _, row in df_sel.iterrows():
            props = {k: (float(row[k]) if k in row and pd.notna(row[k]) else None) for k in ENV_VARS}
            props.update(
                {
                    "id": int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [props["lon"], props["lat"]]},
                    "properties": props,
                }
            )

        return {"type": "FeatureCollection", "features": features}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Serve frontend (must be AFTER /api routes)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
