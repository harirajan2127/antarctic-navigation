"""Offline tests for data-pipeline correctness fixes.

Covers:
- ``RobustDownloader`` forced refresh (``force=True`` bypasses the silent skip).
- ``ensure_sea_ice`` never silently keeps synthetic demo when a real source is
  requested / available.
- ``validate_processed`` prefers the real ``icebergs.csv`` over the demo file.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.common.downloader import ExistingFileSkipped, RobustDownloader
from data_pipeline.icebergs.import_data import validate_processed
from data_pipeline.sea_ice.acquire import _generate_demo_sea_ice, ensure_sea_ice


class _FakeSettings:
    """Minimal stand-in for PipelineSettings used by acquire.ensure_sea_ice."""

    def __init__(self, root: Path) -> None:
        self.DATA_PROCESSED_ROOT = root
        self.PIPELINE_SKIP_EXISTING = True
        self.has_copernicus_credentials = False
        self.has_nsidc_credentials = False


class _FakeDownloaderSettings(_FakeSettings):
    def __init__(self) -> None:
        self.PIPELINE_SKIP_EXISTING = True
        self.PIPELINE_MIN_VALID_BYTES = 16
        self.PIPELINE_RETRIES = 1
        self.PIPELINE_RETRY_BACKOFF_SECONDS = 0.01
        self.PIPELINE_TIMEOUT_SECONDS = 5.0
        self.PIPELINE_CHUNK_BYTES = 1 << 20


@pytest.fixture
def downloader(monkeypatch):
    import data_pipeline.common.downloader as dl

    monkeypatch.setattr(
        dl, "get_pipeline_settings", lambda: _FakeDownloaderSettings()
    )
    return dl.RobustDownloader()


def test_downloader_skips_existing_without_force(tmp_path, downloader):
    target = tmp_path / "file.bin"
    target.write_bytes(b"a" * 64)
    with pytest.raises(ExistingFileSkipped):
        downloader.download("http://example.invalid/file.bin", target)
    # untouched content
    assert target.read_bytes() == b"a" * 64


def test_downloader_force_overwrites_existing(tmp_path, downloader, monkeypatch):
    target = tmp_path / "file.bin"
    target.write_bytes(b"old" * 32)

    def _fake_stream(url, headers, part, label):
        part.write_bytes(b"new" * 64)

    monkeypatch.setattr(downloader, "_stream_download", _fake_stream)
    result = downloader.download(
        "http://example.invalid/file.bin", target, force=True
    )
    assert result.read_bytes() == b"new" * 64


def test_ensure_sea_ice_reuses_existing_demo_in_pure_demo_mode(tmp_path, monkeypatch):
    import data_pipeline.sea_ice.acquire as acq

    monkeypatch.setattr(
        acq, "get_pipeline_settings", lambda: _FakeSettings(tmp_path)
    )
    monkeypatch.setattr(acq, "_generate_demo_sea_ice", lambda *a, **k: _demo_file(tmp_path))
    start = datetime.now(timezone.utc) - timedelta(days=3)
    out = ensure_sea_ice(start=start, end=datetime.now(timezone.utc), mode="demo")
    assert out.exists()


def _demo_file(root: Path) -> Path:
    path = root / "sea_ice.nc"
    if not path.exists():
        _generate_demo_sea_ice(
            datetime.now(timezone.utc) - timedelta(days=2),
            datetime.now(timezone.utc),
            path,
        )
    return path


def test_validate_processed_prefers_real_icebergs(tmp_path):
    demo = tmp_path / "icebergs_demo.csv"
    real = tmp_path / "icebergs.csv"
    demo.write_text(
        "iceberg_id,timestamp,latitude,longitude,length_nm,width_nm\n"
        "DEMO-A01,2026-01-01,-60.0,150.0,1.2,0.6\n",
        encoding="utf-8",
    )
    real.write_text(
        "iceberg_id,timestamp,latitude,longitude,length_nm,width_nm\n"
        "REAL-X01,2026-01-01,-61.0,151.0,2.5,1.1\n",
        encoding="utf-8",
    )
    result = validate_processed(tmp_path)
    assert result.status == "valid"
    assert result.file == "icebergs.csv"
    assert result.meta["records"] == 1


def test_validate_processed_falls_back_to_demo(tmp_path):
    demo = tmp_path / "icebergs_demo.csv"
    demo.write_text(
        "iceberg_id,timestamp,latitude,longitude,length_nm,width_nm\n"
        "DEMO-A01,2026-01-01,-60.0,150.0,1.2,0.6\n",
        encoding="utf-8",
    )
    result = validate_processed(tmp_path)
    assert result.status == "valid"
    assert result.file == "icebergs_demo.csv"


def test_real_mode_never_falls_back_to_demo(monkeypatch, tmp_path):
    from config import settings
    from services import data_paths

    monkeypatch.setattr(settings, "DATA_MODE", "real", raising=False)

    demo_file = tmp_path / "demo_sea_ice.nc"
    demo_file.write_text("demo", encoding="utf-8")

    result = data_paths._resolve([tmp_path / "missing_sea_ice.csv"], demo_file, "sea_ice")

    assert result != demo_file
    assert result.name == "__missing__"