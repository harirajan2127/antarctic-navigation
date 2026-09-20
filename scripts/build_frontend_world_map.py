"""Build a clean, educational full-world equirectangular map texture for the 3D globe.

Renders a 2:1 equirectangular PNG using publicly-available Natural Earth land
polygons (public domain).  Style: soft blue ocean, light cream land, subtle
country borders, thin graticule lines.  Output goes to
`frontend/public/data/world-map.png` (served by Vite as a static asset).

This is ONLY a visual basemap for the 3D-globe UI.  It does not touch the
sea-ice / iceberg data pre-processing or model metrics.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORLD_W = 4096
WORLD_H = 2048

GEOJSON_URL = (
    "https://raw.githubusercontent.com/vasturiano/globe.gl/master/"
    "example/datasets/ne_110m_admin_0_countries.geojson"
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_GEOJSON = ROOT / "Real data" / "external" / "world" / "ne_110m_admin_0_countries.geojson"
OUT_PNG = ROOT / "frontend" / "public" / "data" / "world-map.png"

# Colors (educational / geographic style)
OCEAN_TOP = (163, 204, 235)     # light soft blue (north)
OCEAN_BOT = (141, 186, 222)     # slightly deeper soft blue (south)
LAND = (240, 230, 200)          # light cream / beige
LAND_ANTARCTICA = (241, 240, 230)  # pale near-white for Antarctica
BORDER = (176, 152, 106)        # subtle country border
COAST = (150, 126, 84)          # coastline stroke
GRATICULE = (255, 255, 255)     # thin lat/lon lines
GRATICULE_ALPHA = 90
GRATICULE_EQUATOR_ALPHA = 120
GRATICULE_WIDTH = 1


def lon_lat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0 * WORLD_W
    y = (90.0 - lat) / 180.0 * WORLD_H
    return x, y


def ring_to_points(ring: list) -> list[tuple[float, float]]:
    return [lon_lat_to_xy(lon, lat) for lon, lat in ring]


def ocean_color_at_y(y: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, y / WORLD_H))
    return tuple(int(OCEAN_TOP[i] + (OCEAN_BOT[i] - OCEAN_TOP[i]) * t) for i in range(3))


def download_geojson() -> Path:
    if CACHE_GEOJSON.exists():
        print(f"[world-map] using cached {CACHE_GEOJSON}")
        return CACHE_GEOJSON

    CACHE_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"[world-map] downloading Natural Earth 110m countries -> {CACHE_GEOJSON}")
    req = urllib.request.Request(GEOJSON_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(CACHE_GEOJSON, "wb") as fh:
        fh.write(resp.read())
    print("[world-map] downloaded.")
    return CACHE_GEOJSON


def build_texture() -> None:
    geojson_path = download_geojson()
    with open(geojson_path, "r", encoding="utf-8") as fh:
        geojson = json.load(fh)

    features = geojson.get("features", [])

    # Base ocean gradient canvas
    img = Image.new("RGB", (WORLD_W, WORLD_H))
    draw = ImageDraw.Draw(img)
    for y in range(WORLD_H):
        draw.line([(0, y), (WORLD_W, y)], fill=ocean_color_at_y(y))
    del draw

    overlay = Image.new("RGBA", (WORLD_W, WORLD_H), (0, 0, 0, 0))

    # --- Land polygons (even-odd handling for holes) ------------------------
    land_layer = Image.new("RGBA", (WORLD_W, WORLD_H), (0, 0, 0, 0))
    land_draw = ImageDraw.Draw(land_layer)
    coast_draw = ImageDraw.Draw(land_layer)

    for feature in features:
        props = feature.get("properties", {}) or {}
        name = props.get("NAME") or props.get("name") or ""
        geom = feature.get("geometry") or {}
        gtype = geom.get("type", "")
        is_antarctica = "antarctica" in str(name).lower()

        coords_list = [geom["coordinates"]] if gtype == "Polygon" else geom["coordinates"]
        for poly in coords_list:
            outer = ring_to_points(poly[0])
            fill = LAND_ANTARCTICA if is_antarctica else LAND
            land_draw.polygon(outer, fill=fill)
            coast_draw.line(outer + [outer[0]], fill=COAST, width=1, joint="curve")
            # punch holes back to the ocean gradient color (lakes / exclaves)
            for hole in poly[1:]:
                hpts = ring_to_points(hole)
                cy = hpts[0][1]
                land_draw.polygon(hpts, fill=ocean_color_at_y(cy) + (255,))
                for hy0 in range(int(min(p[1] for p in hpts)), int(max(p[1] for p in hpts)) + 1):
                    land_draw.line(
                        [(hpts[0][0], hy0), (hpts[0][0] + 1, hy0)],
                        fill=ocean_color_at_y(hy0) + (255,),
                    )

        if gtype not in ("Polygon", "MultiPolygon"):
            continue

    # --- Country borders (subtle) -------------------------------------------
    border_draw = ImageDraw.Draw(land_layer)
    for feature in features:
        geom = feature.get("geometry") or {}
        gtype = geom.get("type", "")
        if gtype in ("Polygon", "MultiPolygon"):
            coords_list = [geom["coordinates"]] if gtype == "Polygon" else geom["coordinates"]
            for poly in coords_list:
                border_draw.line(ring_to_points(poly[0]) + [ring_to_points(poly[0])[0]],
                                 fill=BORDER, width=1, joint="curve")

    # --- Graticule (thin lat/lon lines) --------------------------------------
    grat_draw = ImageDraw.Draw(overlay)
    for lon in range(-180, 180, 15):
        x0, y0 = lon_lat_to_xy(lon + 0.0, -90)
        x1, y1 = lon_lat_to_xy(lon + 0.0, 90)
        grat_draw.line([(x0, y0), (x1, y1)], fill=GRATICULE + (GRATICULE_ALPHA,), width=GRATICULE_WIDTH)
    for lat in [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]:
        x0, y0 = lon_lat_to_xy(-180, lat + 0.0)
        x1, y1 = lon_lat_to_xy(180, lat + 0.0)
        alpha = GRATICULE_EQUATOR_ALPHA if lat == 0 else GRATICULE_ALPHA
        grat_draw.line([(x0, y0), (x1, y1)], fill=GRATICULE + (alpha,), width=GRATICULE_WIDTH)

    # Compose: ocean -> graticule-under -> land -> graticule-over(faint)
    base_rgba = img.convert("RGBA")
    composed = Image.alpha_composite(base_rgba, overlay)
    composed = Image.alpha_composite(composed, land_layer)
    composed = Image.alpha_composite(composed, overlay)

    # Slight top highlight so sphere reads as rounded
    edge = Image.new("RGBA", (WORLD_W, WORLD_H), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge)
    # subtle light band at the top of the sphere (north pole area)
    edge_draw.rectangle([(0, 0), (WORLD_W, int(WORLD_H * 0.08))], fill=(255, 255, 255, 22))
    composed = Image.alpha_composite(composed, edge)

    final = composed.convert("RGB")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUT_PNG, format="PNG", optimize=True)
    size_kb = OUT_PNG.stat().st_size / 1024
    print(f"[world-map] wrote {OUT_PNG} ({size_kb:.0f} KB, {final.size[0]}x{final.size[1]})")


if __name__ == "__main__":
    sys.exit(build_texture())