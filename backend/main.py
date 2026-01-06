from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from shapely.geometry import shape, Point
from shapely.prepared import prep
from pathlib import Path
from pyproj import Transformer

from backend.soil_sampling_engine import suggest_clhs_samples

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "acre_points_small.csv"
FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

df_all = pd.read_csv(DATA_PATH)

# Convert UTM 16N -> lon/lat ONCE at startup
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

@app.get("/")
def home():
    # Serve the frontend on Render at "/"
    if not INDEX_HTML.exists():
        return JSONResponse(status_code=500, content={"error": f"Missing {INDEX_HTML} on server"})
    return FileResponse(INDEX_HTML)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/extent")
def extent():
    # Leaflet bounds format: [[southLat, westLon], [northLat, eastLon]]
    return {"bounds": [[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]]}

def _make_valid_polygon(g):
    # Try make_valid if available; otherwise buffer(0)
    try:
        from shapely.validation import make_valid
        return make_valid(g)
    except Exception:
        return g.buffer(0)

@app.post("/api/suggest-samples")
def suggest_samples(req: SamplingRequest):
    try:
        poly_ll = shape(req.polygon)
        poly_ll = _make_valid_polygon(poly_ll)

        if poly_ll.is_empty:
            return {"type": "FeatureCollection", "features": []}

        # Quick bbox reject (prevents "random hits" if drawing far away)
        minx, miny, maxx, maxy = poly_ll.bounds  # lon/lat
        if (maxx < MIN_LON) or (minx > MAX_LON) or (maxy < MIN_LAT) or (miny > MAX_LAT):
            return {"type": "FeatureCollection", "features": []}

        prepared = prep(poly_ll)

        mask = [
            prepared.covers(Point(lon, lat))  # includes boundary
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
        return JSONResponse(status_code=500, content={"error": str(e)})
