from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import time
import hashlib

import pandas as pd
from shapely.geometry import shape, Point
from shapely.prepared import prep
from pathlib import Path
from pyproj import Transformer
from fastapi.staticfiles import StaticFiles

from backend.soil_sampling_engine import (
    suggest_clhs_samples,
    recommend_sample_sizes,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "acre_points_small.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"

df_all = pd.read_csv(DATA_PATH)

# Convert UTM 16N -> lon/lat ONCE
to_ll = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True).transform
lonlat = [to_ll(float(x), float(y)) for x, y in df_all[["x", "y"]].to_numpy()]
df_all["lon"] = [p[0] for p in lonlat]
df_all["lat"] = [p[1] for p in lonlat]

MIN_LON, MAX_LON = float(df_all["lon"].min()), float(df_all["lon"].max())
MIN_LAT, MAX_LAT = float(df_all["lat"].min()), float(df_all["lat"].max())

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


def _make_valid_polygon(g):
    try:
        from shapely.validation import make_valid
        return make_valid(g)
    except Exception:
        return g.buffer(0)


def _polygon_candidates(poly_geojson: Dict[str, Any]) -> pd.DataFrame:
    poly_ll = shape(poly_geojson)
    poly_ll = _make_valid_polygon(poly_ll)

    if poly_ll.is_empty:
        return df_all.iloc[0:0].copy()

    # Quick bbox reject
    minx, miny, maxx, maxy = poly_ll.bounds  # lon/lat
    if (maxx < MIN_LON) or (minx > MAX_LON) or (maxy < MIN_LAT) or (miny > MAX_LAT):
        return df_all.iloc[0:0].copy()

    prepared = prep(poly_ll)
    mask = [prepared.covers(Point(lon, lat)) for lon, lat in zip(df_all["lon"], df_all["lat"])]
    return df_all.loc[mask].copy()


# small cache for /api/recommend-n (keeps Render snappy)
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

        df_sub = _polygon_candidates(req.polygon)
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
        df_sub = _polygon_candidates(req.polygon)
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
            props = {
                "id": int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
                # return both lon/lat and original x/y so user can use either
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "NDVI": float(row.get("NDVI", 0)),
                "total_clay": float(row.get("total_clay", 0)),
                "slope": float(row.get("slope", 0)),
            }
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [props["lon"], props["lat"]]},
                "properties": props,
            })

        return {"type": "FeatureCollection", "features": features}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Serve frontend (must be AFTER /api routes)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
