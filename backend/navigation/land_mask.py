"""Land-mask loading for the Antarctic routing grid.

The navigation engine needs to know which cells are land (Antarctica and
islands) so routes cannot cross the continent. This module loads a coastline /
bathymetry layer from disk and maps it onto a :class:`AntarcticGrid`.

Supported formats (configurable via ``LAND_MASK_FILE``):

* NetCDF with an ``elevation`` variable (GEBCO-style: land where
  ``elevation >= 0``) or a ``depth`` variable (land where ``depth < 0``),
* a raw ``.npy`` boolean array already aligned to the grid,
* ``.tif``/GeoTIFF rasters via rasterio when installed (elevation >= 0 = land), and
* ``.json`` produced by the real-data pipeline: parallel ``latitude``/``longitude``
  coordinate-pair arrays marking every ocean cell (everything else is land).

If no file is configured or the file cannot be read, :func:`load_land_mask`
returns ``(None, info)`` — the caller must then keep the whole region as open
water and surface this as an explicit limitation (never silently).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from navigation.grid import AntarcticGrid


def _land_from_elevation(lat, lon, elevation) -> np.ndarray:
    flat_lat = np.sort(np.unique(np.asarray(lat, dtype=float)))
    flat_lon = np.sort(np.unique(np.asarray(lon, dtype=float)))
    elev = np.asarray(elevation, dtype=float)
    if elev.ndim == 1 and len(elev) == len(flat_lat) * len(flat_lon):
        elev = elev.reshape(len(flat_lat), len(flat_lon))
    if elev.shape != (len(flat_lat), len(flat_lon)):
        lat2d, lon2d = np.meshgrid(flat_lat, flat_lon, indexing="ij")
        from scipy.interpolate import griddata

        points = np.column_stack([np.asarray(lat).ravel(), np.asarray(lon).ravel()])
        elev = griddata(points, elev.ravel(), (lat2d.ravel(), lon2d.ravel()),
                        method="nearest").reshape(lat2d.shape)
    return np.where(np.isfinite(elev), elev >= 0, False)


def load_land_mask(grid: AntarcticGrid, path: str | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load and regrid a land mask onto ``grid``.

    Returns ``(mask, info)``; ``mask`` is ``None`` when no usable land layer is
    available. ``info`` always describes exactly what was (or was not) loaded.
    """
    if not path:
        return None, {
            "loaded": False,
            "file": None,
            "reason": "LAND_MASK_FILE is not set; the grid is treated as open water.",
        }

    full = Path(path)
    if not full.exists():
        return None, {
            "loaded": False,
            "file": str(full),
            "reason": f"Configured land-mask file does not exist: {full}",
        }

    src: dict[str, Any] = {}
    try:
        suffix = full.suffix.lower()
        if suffix == ".npy":
            mask = np.load(full)
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (grid.nlat, grid.nlon):
                raise ValueError(
                    f"Land mask shape {mask.shape} does not match grid "
                    f"{(grid.nlat, grid.nlon)} (npy masks must be pre-aligned)."
                )
            masked = mask
            src["format"] = "npy"
            info = {
                "loaded": True,
                "file": str(full),
                "format": "npy",
                "land_cells": int(mask.sum()),
                "land_fraction": float(mask.mean()),
            }
            return masked, info
        elif suffix in (".tif", ".tiff"):
            import rasterio

            with rasterio.open(full) as ds:
                band = ds.read(1)
                transform = ds.transform
                rows, cols = np.indices(band.shape)
                xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
                lat = np.asarray([float(y) for y in ys.ravel().tolist()])
                lon = np.asarray([float(x) for x in xs.ravel().tolist()])
            land = np.where(np.isfinite(band), band >= 0, False)
            src["format"] = "geotiff"
        elif suffix == ".json":
            import ijson

            with full.open("rb") as handle:
                o_lat = np.fromiter(ijson.items(handle, "latitude.item"), dtype=float)
            with full.open("rb") as handle:
                o_lon = np.fromiter(ijson.items(handle, "longitude.item"), dtype=float)
            if len(o_lat) != len(o_lon):
                raise ValueError("JSON land mask latitude/longitude arrays must be equal length.")
            lat_arr = np.sort(np.unique(o_lat))
            lon_arr = np.sort(np.unique(o_lon))
            lat_idx = np.unique(o_lat, return_inverse=True)[1]
            lon_idx = np.unique(o_lon, return_inverse=True)[1]
            land = np.ones((len(lat_arr), len(lon_arr)), dtype=bool)
            land[lat_idx, lon_idx] = False
            lat, lon = o_lat, o_lon
            del lat_idx, lon_idx
            src["format"] = "json"
        else:  # netCDF (GEBCO-style or depth variable)
            import xarray as xr

            with xr.open_dataset(full) as ds:
                lat = ds["lat"].values if "lat" in ds else ds["latitude"].values
                lon = ds["lon"].values if "lon" in ds else ds["longitude"].values
                if "elevation" in ds:
                    elev = ds["elevation"].values
                    land = _land_from_elevation(lat, lon, elev)
                elif "depth" in ds:
                    land = _land_from_elevation(lat, lon, -np.asarray(ds["depth"].values))
                else:
                    raise ValueError(
                        "NetCDF land mask must contain an 'elevation' or 'depth' variable."
                    )
            src["format"] = "netcdf"
    except Exception as exc:  # noqa: BLE001 - report, never raise into routing
        return None, {
            "loaded": False,
            "file": str(full),
            "reason": f"Failed to read land-mask file: {exc}",
        }

    lat_arr = np.sort(np.unique(np.asarray(lat, dtype=float).ravel()))
    lon_arr = np.sort(np.unique(np.asarray(lon, dtype=float).ravel()))
    land2d = np.asarray(land, dtype=bool).reshape(len(lat_arr), len(lon_arr))

    lat_idx = np.array([int(np.argmin(np.abs(lat_arr - lat))) for lat in grid.lats])
    lon_idx = np.array([int(np.argmin(np.abs(lon_arr - lon))) for lon in grid.lons])
    mask = land2d[np.ix_(lat_idx, lon_idx)]

    info = {
        "loaded": True,
        "file": str(full),
        "format": src.get("format", "unknown"),
        "land_cells": int(mask.sum()),
        "land_fraction": float(mask.mean()),
    }
    return mask, info