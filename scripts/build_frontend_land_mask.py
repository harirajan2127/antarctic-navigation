#!/usr/bin/env python3
"""Build a compact, real-data-derived land mask asset for the frontend globe.

Reads the production land/ocean mask (land_ocean_mask.json, ocean-cell
coordinates only) and downsamples it to a coarse grid that ships with the
frontend (frontend/public/data/land_mask.json).  The frontend globe uses this
small file to keep its sea-ice / iceberg layers ocean-only.

Output schema:
{
  "step_deg": 0.1,
  "lat_start": -70.4, "lat_end": -50.0, "n_lat": N,
  "lon_start": 20.0, "lon_end": 120.0, "n_lon": M,
  "land": [0|1, ...],        # row-major over lat (row 0 = lat_start)
  "ocean_cells": int, "land_cells": int,
  "generated_at": "...", "source": "..."
}
"""
import json
import math
import sys
import time
from pathlib import Path

STEP = 0.1
SOURCE = Path("Real data/processed/bathymetry/land_ocean_mask.json")
OUT = Path("frontend/public/data/land_mask.json")


def main() -> int:
    with SOURCE.open() as f:
        m = json.load(f)

    lat = m["latitude"]
    lon = m["longitude"]

    lat_min, lat_max = math.floor(min(lat) / STEP) * STEP, -50.0
    lon_min, lon_max = 20.0, 120.0  # aligned so lon_min % STEP == 0
    n_lat = round((lat_max - lat_min) / STEP) + 1
    n_lon = round((lon_max - lon_min) / STEP) + 1
    n_lat = max(n_lat, 0)
    n_lon = max(n_lon, 0)

    # 1 = land (ocean cells absent from mask), 0 = ocean.
    land = [1] * (n_lat * n_lon)

    # Cell width never straddles the mask boundary badly at 0.1*.
    for li, lng in zip(lat, lon):
        if li < lat_min or li > lat_max or lng < lon_min or lng > lon_max:
            continue
        c = round((lng - lon_min) / STEP)
        r = round((lat_max - li) / STEP)  # row 0 = highest lat (north)
        if 0 <= r < n_lat and 0 <= c < n_lon:
            land[r * n_lon + c] = 0

    land_cells = sum(land)
    ocean_cells = len(land) - land_cells

    out = {
        "step_deg": STEP,
        "lat_start": lat_min,
        "lat_end": lat_max,
        "n_lat": n_lat,
        "lon_start": lon_min,
        "lon_end": lon_max,
        "n_lon": n_lon,
        "land": land,
        "ocean_cells": ocean_cells,
        "land_cells": land_cells,
        "total_cells": len(land),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(SOURCE),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(
        f"wrote {OUT} ({size_kb:.0f} KB), "
        f"ocean={ocean_cells} land={land_cells} of {len(land)}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())