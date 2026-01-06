from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from shapely.geometry import shape, Point
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

from backend.soil_sampling_engine import suggest_clhs_samples

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "acre_points_small.csv"
df_all = pd.read_csv(DATA_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



from shapely.ops import transform
from pyproj import Transformer

# lon/lat -> UTM 16N
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True).transform
# UTM 16N -> lon/lat  ✅ add this
to_ll = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True).transform
class SamplingRequest(BaseModel):
    polygon: Dict[str, Any]  # GeoJSON polygon
    n_samples: int = 30

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/suggest-samples")
def suggest_samples(req: SamplingRequest):
    # 1) Build polygon geometry from GeoJSON
    poly = shape(req.polygon)

    poly_ll = shape(req.polygon)           # polygon in lon/lat
    poly = transform(to_utm, poly_ll)      # polygon in UTM meters (matches your CSV)
    
    mask = df_all.apply(lambda r: poly.contains(Point(r["x"], r["y"])), axis=1)
    df_sub = df_all[mask]



    if df_sub.empty:
        return {"error": "No candidate points inside polygon."}

    if len(df_sub) < req.n_samples:
        n_samples = len(df_sub)
    else:
        n_samples = req.n_samples

    # 3) CLHS on subset
    df_sel = suggest_clhs_samples(
        df_sub,
        n_samples=n_samples,
        polygon_geojson=req.polygon,   # for stable seed
        include_xy_in_clhs=False,      # keep False initially
        scale_mode="rank_normal"       # matches your analysis choices
    )
    # 4) Return as GeoJSON FeatureCollection
    features = []
    for _, row in df_sel.iterrows():
        lon, lat = to_ll(float(row["x"]), float(row["y"]))  # ✅ convert back
    
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},  # ✅ lon/lat
            "properties": {
                "id": int(row["id"]) if "id" in row else None,
                "slope": float(row.get("slope", 0)),
                "NDVI": float(row.get("NDVI", 0)),
                "total_clay": float(row.get("total_clay", 0)),
            },
        })

    return {"type": "FeatureCollection", "features": features}


from fastapi.staticfiles import StaticFiles
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

# Put this AFTER defining /api routes, so /api stays working
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")