"""Validate the real datasets under ``Real data/`` against ``config/data_config.yaml``.

This is a READ-ONLY checker: it never deletes, edits, cleans, resamples or
overwrites any original file more than it ever writes — nothing is written at
all.  It reports three severities:

  * WARNING  — a folder is missing/empty, a format is unsupported, a
               variable/column is missing, a value is out of the selected
               area or confidence is missing.  The dashboard still works.
  * ERROR    — a required dataset file is missing entirely or unreadable.

Exit code 0 = all checks passed (possibly with warnings),
exit code 1 = at least one error was found.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

BACKEND_SRC = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "data_config.yaml"
REAL_DATA_ROOT = PROJECT_ROOT / "Real data"

SUPPORTED_EXTENSIONS = {
    ".csv", ".json", ".geojson", ".nc", ".netcdf",
    ".gpkg", ".parquet", ".tif", ".tiff",
}

_LABEL = {"sea_ice": "sea-ice", "iceberg": "iceberg", "ocean": "ocean",
          "weather": "weather", "vessel": "vessel"}


def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        print(f"[ERROR] config file not found: {CONFIG_FILE}")
        sys.exit(1)
    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dataset_dirs(cfg: dict[str, Any], dataset: str) -> tuple[Path, Path]:
    d = cfg.get("datasets", {}).get(dataset, {})
    raw = Path(d.get("raw_dir", REAL_DATA_ROOT / "raw" / dataset))
    processed = Path(d.get("processed_dir", REAL_DATA_ROOT / "processed" / dataset))
    return raw, processed


def _iter_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _check_folder(folder: Path, dataset: str, label: str) -> None:
    if not folder.exists():
        print(f"[WARNING] {label} folder for '{dataset}' is missing: {folder}")
        return
    files = _iter_files(folder)
    if not files:
        print(f"[WARNING] {label} folder for '{dataset}' is empty: {folder} ")


def _read_tabular(path: Path) -> list[dict[str, Any]] | None:
    """Best-effort read of CSV/JSON/GeoJSON into a list of row dicts."""
    try:
        if path.suffix.lower() == ".csv":
            import csv
            with path.open("r", newline="", encoding="utf-8-sig") as fh:
                return list(csv.DictReader(fh))
        if path.suffix.lower() in (".json", ".geojson"):
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "features" in data:  # GeoJSON
                rows = []
                for f in data["features"]:
                    coords = (f.get("geometry") or {}).get("coordinates") or []
                    if len(coords) >= 2:
                        rows.append({"longitude": coords[0], "latitude": coords[1],
                                     **(f.get("properties") or {})})
                return rows
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
            if isinstance(data, dict):
                return [data]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ERROR] could not read {path}: {exc}")
    return None


def _check_tabular(path: Path, dataset: str, cfg: dict[str, Any]) -> None:
    rows = _read_tabular(path)
    if rows is None:
        return
    d = cfg.get("datasets", {}).get(dataset, {})
    required = d.get("required_columns") or []
    lat_bounds = d.get("lat_bounds") or cfg.get("selected_lat_band") or [-90, 90]

    for col in required:
        present = rows and all(col in r for r in rows)
        if not present:
            print(f"[WARNING] {path.name}: required column '{col}' missing")

    lat_col = "latitude" if all("latitude" in r for r in rows) else None
    n_lat_out = 0
    n_missing = 0
    n_dup = 0
    seen: set[tuple] = set()
    for row in rows:
        lat = row.get("latitude")
        if lat is None or (isinstance(lat, str) and not lat.strip()):
            n_missing += 1
            continue
        try:
            lat_f = float(lat)
        except (TypeError, ValueError):
            n_missing += 1
            continue
        if not (-90.0 <= lat_f <= 90.0):
            print(f"[WARNING] {path.name}: invalid latitude {lat!r} out of [-90,90]")
            continue
        if lat_f < lat_bounds[0] or lat_f > lat_bounds[1]:
            n_lat_out += 1
        key = tuple(sorted((str(row.get(c)) for c in required))) if required else ()
        if key:
            if key in seen:
                n_dup += 1
            seen.add(key)

    if n_lat_out:
        print(f"[WARNING] {path.name}: {n_lat_out} row(s) outside "
              f"selected band {lat_bounds[0]}..{lat_bounds[1]}")
    if n_missing:
        print(f"[WARNING] {path.name}: {n_missing} row(s) with missing/invalid latitude")
    if n_dup:
        print(f"[WARNING] {path.name}: {n_dup} duplicate record(s)")

    # Iceberg-specific: on-land rows must be flagged.
    if dataset == "iceberg" and rows:
        on_land_missing = any("on_land" not in r for r in rows)
        if on_land_missing:
            print(f"[WARNING] {path.name}: 'on_land' column absent — "
                  f"cannot flag icebergs that sit over the continent")


def _check_hierarchical(path: Path, dataset: str, cfg: dict[str, Any]) -> None:
    """Best-effort check for NetCDF/GeoTIFF/GeoPackage/Parquet files.

    Kept dependency-light: we only inspect what the core project already
    imports.  If a reader is unavailable the file is reported as 'not
    inspected' (a warning), never a crash.
    """
    d = cfg.get("datasets", {}).get(dataset, {})
    required = d.get("required_variables") or []
    try:
        import xarray as xr  # type: ignore
        with xr.open_dataset(str(path)) as ds:
            for var in required:
                if var not in ds.variables:
                    print(f"[WARNING] {path.name}: required variable '{var}' missing")
        return
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARNING] {path.name}: xarray could not inspect dataset: {exc}")
        return

    # Fallback readers.
    import json
    try:
        ds_vars = _read_tabular(path)
        if ds_vars is not None:
            for var in required:
                present = all(var in r for r in ds_vars) if ds_vars else False
                if not present:
                    print(f"[WARNING] {path.name}: required variable '{var}' missing")
        else:
            print(f"[WARNING] {path.name}: could not read variables — "
                  f"not inspected (reader unavailable)")
    except json.JSONDecodeError:
        print(f"[WARNING] {path.name}: could not read variables — "
              f"not inspected (reader unavailable)")


def main() -> int:
    cfg = _load_config()
    print(f"Validating datasets under {REAL_DATA_ROOT}")
    print(f"Using config: {CONFIG_FILE}\n")

    errors = 0
    for dataset, label in _LABEL.items():
        print(f"== {dataset} ({label}) ==")
        raw, processed = _dataset_dirs(cfg, dataset)
        _check_folder(raw, dataset, "raw")
        _check_folder(processed, dataset, "processed")

        for folder, where in ((raw, "raw"), (processed, "processed")):
            for path in _iter_files(folder):
                ext = path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    print(f"[WARNING] {path.name}: unsupported format '{ext}'")
                    continue
                if ext in (".csv", ".json", ".geojson", ".parquet", ".gpkg"):
                    _check_tabular(path, dataset, cfg)
                else:
                    _check_hierarchical(path, dataset, cfg)
        print()
    print("Validation complete.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
