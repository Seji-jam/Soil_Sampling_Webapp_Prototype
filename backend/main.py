from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from shapely.geometry import shape, Point
from shapely.prepared import prep
from pathlib import Path
from pyproj import Transformer

from backend.soil_sampling_engine import suggest_clhs_samples

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "acre_points_small.csv"
df_all = pd.read_csv(DATA_PATH)

# Convert UTM16N -> lon/lat ONCE
to_ll = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True).transform
lons_lats = [to_ll(x, y) for x, y in df_all[["x", "y"]].to_numpy()]
df_all["lon"] = [p[0] for p in lons_lats]
df_all["lat"] = [p[1] for p in lons_lats]

# Dataset extent in lon/lat
MIN_LON, MAX_LON = float(df_all["lon"].min()), float(df_all["lon"].max())
MIN_LAT, MAX_LAT = float(df_all["lat"].min()), float(df_all["lat"].max())

app = FastAPI()

class SamplingRequest(BaseModel):
    polygon: Dict[str, Any]  # GeoJSON geometry (Polygon)
    n_samples: int = 30

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/extent")
def extent():
    # Leaflet expects: [[southWestLat, southWestLng], [northEastLat, northEastLng]]
    return {"bounds": [[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]]}

def _make_valid_polygon(g):
    # Shapely 2 has make_valid in shapely.validation; older versions don't.
    try:
        from shapely.validation import make_valid
        return make_valid(g)
    except Exception:
        # buffer(0) is a common fix for self-intersections
        return g.buffer(0)

@app.post("/api/suggest-samples")
def suggest_samples(req: SamplingRequest):
    try:
        poly_ll = shape(req.polygon)
        poly_ll = _make_valid_polygon(poly_ll)

        if poly_ll.is_empty:
            return {"type": "FeatureCollection", "features": []}

        # Quick bbox reject: if polygon is totally outside dataset bbox, return empty.
        minx, miny, maxx, maxy = poly_ll.bounds  # lon/lat bounds
        if (maxx < MIN_LON) or (minx > MAX_LON) or (maxy < MIN_LAT) or (miny > MAX_LAT):
            return {"type": "FeatureCollection", "features": []}

        prepared = prep(poly_ll)

        # Point-in-polygon in lon/lat (THIS prevents false hits anywhere else on the globe)
        mask = [
            prepared.covers(Point(lon, lat))   # covers includes boundary points
            for lon, lat in zip(df_all["lon"], df_all["lat"])
        ]
        df_sub = df_all.loc[mask].copy()

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
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                "properties": {
                    "id": int(row["id"]) if "id" in row else None,
                    "NDVI": float(row.get("NDVI", 0)),
                    "total_clay": float(row.get("total_clay", 0)),
                    "slope": float(row.get("slope", 0)),
                },
            })

        return {"type": "FeatureCollection", "features": features}

    except Exception as e:
        # Always return JSON so the frontend can display a useful error
        return JSONResponse(status_code=500, content={"error": str(e)})
