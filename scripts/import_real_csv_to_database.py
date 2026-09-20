#!/usr/bin/env python3
"""Migrate the DSS database from demo data to real processed data.

Actions:
1. Back up the existing database.
2. Remove every demo-only record (``demo=True``) from all tables.
3. Import real iceberg observations from the processed CSV.
4. Upsert dataset_metadata with real-* source labels and live record counts.
5. Insert sentinel model_metrics marking ``Not Trained`` (demo=False, trained_on=None)
   so the analytics summary no longer forces demo_mode=True.
6. Preserve the polar_explorer vessel config used by the routing engine.

Exit code 0 = success, 1 = any step failed.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DB_PATH = BACKEND_DIR / "antarctic_dss.db"
BACKUP_PATH = BACKEND_DIR / "antarctic_dss_backup.db"

# Pin the database path BEFORE importing the backend so the migration always
# touches backend/antarctic_dss.db regardless of the process CWD.
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"
ICEBERG_CSV = (
    PROJECT_ROOT / "Real data" / "processed" / "iceberg" / "csv" / "iceberg_processed.csv"
)
WEATHER_ROOT = PROJECT_ROOT / "Real data" / "processed" / "weather" / "csv"
SEA_ICE_ROOT = PROJECT_ROOT / "Real data" / "processed" / "sea_ice" / "csv"
OCEAN_ROOT = PROJECT_ROOT / "Real data" / "processed" / "ocean" / "csv"
VESSEL_CSV = PROJECT_ROOT / "Real data" / "processed" / "vessel" / "csv" / "vessel_processed.csv"
BATHY_DIR = PROJECT_ROOT / "Real data" / "processed" / "bathymetry" / "csv"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _banner(msg: str) -> None:
    print(f"\n{'='*64}\n  [{_ts()}] {msg}\n{'='*64}", flush=True)


def _backup_db() -> bool:
    _banner("Backup database")
    if not DB_PATH.exists():
        print("  No existing database to back up.")
        return True
    try:
        shutil.copy2(str(DB_PATH), str(BACKUP_PATH))
        print(f"  Backed up: {DB_PATH} -> {BACKUP_PATH} ({DB_PATH.stat().st_size:,} bytes)")
        return True
    except Exception as exc:
        print(f"  BACKUP FAILED: {exc}")
        return False


def _clear_demo_records() -> bool:
    _banner("Remove demo-only records")
    try:
        from database.database import engine, SessionLocal
        from database.models import Base
        from sqlalchemy import text

        Base.metadata.create_all(bind=engine)

        tables_with_demo = [
            "icebergs", "iceberg_observations", "sea_ice_forecasts",
            "routes", "model_metrics",
        ]
        with SessionLocal() as db:
            for table in tables_with_demo:
                result = db.execute(text(f"DELETE FROM {table} WHERE demo = :d"), {"d": 1})
                if result.rowcount:
                    print(f"  {table}: deleted {result.rowcount} demo rows")

            # Dataset metadata: delete and re-insert (will be rebuilt below)
            db.execute(text("DELETE FROM dataset_metadata"))
            print("  dataset_metadata: cleared for rebuild")
            db.commit()
        return True
    except Exception as exc:
        print(f"  CLEAR FAILED: {exc}")
        return False


def _count_csv_rows(path: Path) -> int:
    """Fast binary line count (minus 1 for header)."""
    n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


def _import_icebergs() -> bool:
    _banner("Import real iceberg observations")
    if not ICEBERG_CSV.exists():
        print(f"  SKIP: {ICEBERG_CSV} not found")
        return True
    try:
        from database.database import SessionLocal
        from database.models import Iceberg, IcebergObservation, utcnow

        rows = []
        with ICEBERG_CSV.open("r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)

        print(f"  Read {len(rows)} observations from {ICEBERG_CSV.name}")

        # Build per-iceberg latest position for the icebergs table.
        latest: dict[str, dict] = {}
        observations: list[dict] = []
        for row in rows:
            iid = row["iceberg_id"]
            obs = {
                "iceberg_id": iid,
                "timestamp": datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "source": f"real_csv:{row.get('source_file', '')}",
                "demo": False,
            }
            observations.append(obs)
            if iid not in latest or obs["timestamp"] > latest[iid]["timestamp"]:
                latest[iid] = obs

        with SessionLocal() as db:
            for iid, pos in latest.items():
                length_nm = None
                width_nm = None
                remarks = ""
                for row in rows:
                    if row["iceberg_id"] == iid:
                        try:
                            length_nm = float(row["length_nm"]) if row.get("length_nm") else None
                        except (TypeError, ValueError):
                            pass
                        try:
                            width_nm = float(row["width_nm"]) if row.get("width_nm") else None
                        except (TypeError, ValueError):
                            pass
                        remarks = row.get("remarks", "") or ""
                        break
                db.merge(Iceberg(
                    iceberg_id=iid,
                    name=remarks,
                    latitude=pos["latitude"],
                    longitude=pos["longitude"],
                    length_km=length_nm * 1.852 if length_nm else None,
                    width_km=width_nm * 1.852 if width_nm else None,
                    last_observed=pos["timestamp"],
                    source="real_csv",
                    classification="pipeline_data",
                    demo=False,
                ))

            db.commit()

            # Bulk insert observations in batches.
            BATCH = 5000
            for start in range(0, len(observations), BATCH):
                batch = observations[start:start + BATCH]
                db.bulk_insert_mappings(IcebergObservation, batch)
            db.commit()

        print(f"  Upserted {len(latest)} icebergs, inserted {len(observations)} observations")
        return True
    except Exception as exc:
        print(f"  ICEBERG IMPORT FAILED: {exc}")
        return False


def _count_real_records() -> dict[str, int | None]:
    """Count real processed records per dataset (None = grid data, no row count)."""
    counts: dict[str, int | None] = {}
    # Sea-ice: total rows in the latest part
    si_dir = SEA_ICE_ROOT
    if si_dir.is_dir():
        latest = sorted(si_dir.glob("sea_ice_latest_part_*.csv"))
        if latest:
            counts["sea_ice"] = _count_csv_rows(latest[-1])
        else:
            counts["sea_ice"] = None
    else:
        counts["sea_ice"] = None

    # Ocean/weather: all parts
    for name, root in [("ocean", OCEAN_ROOT), ("weather", WEATHER_ROOT)]:
        if root.is_dir():
            parts = sorted(root.glob(f"{name}_part_*.csv"))
            if parts:
                total = sum(_count_csv_rows(p) for p in parts)
                counts[name] = total
            else:
                counts[name] = None
        else:
            counts[name] = None

    # Vessel
    if VESSEL_CSV.exists():
        counts["vessel"] = _count_csv_rows(VESSEL_CSV)
    else:
        counts["vessel"] = None

    # Bathymetry
    if BATHY_DIR.is_dir():
        parts = sorted(BATHY_DIR.glob("bathymetry_part_*.csv"))
        counts["bathymetry"] = sum(_count_csv_rows(p) for p in parts) if parts else None
    else:
        counts["bathymetry"] = None

    return counts


def _update_dataset_metadata() -> bool:
    _banner("Update dataset metadata with real-* labels")
    try:
        from database.database import SessionLocal
        from database.models import DatasetRecord, utcnow

        records = _count_real_records()
        print(f"  Record counts: {records}")

        with SessionLocal() as db:
            metadata = [
                ("sea_ice", "real_sea_ice", "Real Sea-Ice Concentration", "ok",
                 records.get("sea_ice"), "processed from NSIDC sea-ice CSV"),
                ("iceberg", "real_iceberg", "Real Iceberg Observations", "ok",
                 records.get("iceberg"), "processed from Antarctic iceberg CSV"),
                ("weather", "real_weather", "Real Surface Weather", "ok",
                 records.get("weather"), "processed from ERA5 grib data"),
                ("ocean_currents", "real_ocean", "Real Ocean Temperature/Currents", "ok",
                 records.get("ocean"), "processed from Copernicus NetCDF"),
                ("vessel", "real_vessel", "Real Vessel AIS (fleet aggregate)", "ok",
                 records.get("vessel"), "processed from Global Fishing Watch AIS"),
                ("bathymetry", "real_bathymetry", "Real Bathymetry/Elevation", "ok",
                 records.get("bathymetry"), "processed from GEBCO/IBCSO NetCDF"),
            ]

            for ds_name, classification, source_name, status, recs, notes in metadata:
                db.add(DatasetRecord(
                    dataset_name=ds_name,
                    classification=classification,
                    source_name=source_name,
                    status=status,
                    records=recs,
                    demo=False,
                    notes=notes,
                    updated_at=utcnow(),
                ))
            db.commit()

        print(f"  Upserted {len(metadata)} dataset metadata records")
        return True
    except Exception as exc:
        print(f"  METADATA FAILED: {exc}")
        return False


def _insert_model_sentinels() -> bool:
    _banner("Insert model sentinel metrics (Not Trained)")
    try:
        from database.database import SessionLocal
        from database.models import ModelMetric, utcnow

        sentinels = [
            {
                "pipeline": "sea_ice",
                "model": "persistence",
                "metric_name": "status",
                "metric_value": 0.0,
                "split": None,
                "demo": False,
                "trained_on": None,
                "source_file": "import_real_csv_to_database.py",
            },
            {
                "pipeline": "iceberg",
                "model": "persistence",
                "metric_name": "status",
                "metric_value": 0.0,
                "split": None,
                "demo": False,
                "trained_on": None,
                "source_file": "import_real_csv_to_database.py",
            },
        ]
        with SessionLocal() as db:
            for s in sentinels:
                db.add(ModelMetric(**s, created_at=utcnow()))
            db.commit()
        print("  Inserted 2 sentinel metrics (demo=False, trained_on=None)")
        return True
    except Exception as exc:
        print(f"  SENTINEL FAILED: {exc}")
        return False


def main() -> int:
    print(f"  Migration script: {_ts()}")
    print(f"  Database: {DB_PATH}")

    if not _backup_db():
        return 1
    if not _clear_demo_records():
        return 1
    if not _import_icebergs():
        return 1
    if not _update_dataset_metadata():
        return 1
    if not _insert_model_sentinels():
        return 1

    _banner("MIGRATION COMPLETE")
    print(f"  Database size: {DB_PATH.stat().st_size:,} bytes")
    print(f"  Backup: {BACKUP_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
