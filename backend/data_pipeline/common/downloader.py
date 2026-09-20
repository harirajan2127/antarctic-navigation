"""Robust file downloader.

Features:
- Retries with backoff for transient failures.
- Detects incomplete files (Content-Length mismatch).
- Optional SHA-256 verification.
- Does not redownload existing valid files (``PIPELINE_SKIP_EXISTING``).
- Downloads to a ``.part`` file first, then atomically renames.
- Logs progress.
"""
from __future__ import annotations

import hashlib
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger

log = get_logger(__name__)


class DownloadError(Exception):
    """Raised when a download cannot be completed after retries."""


class ExistingFileSkipped(Exception):
    """Raised when the target already exists and is considered valid."""


def _sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class RobustDownloader:
    """URL-to-file downloader with retry and integrity checks."""

    def __init__(self) -> None:
        self.settings = get_pipeline_settings()

    def download(
        self,
        url: str,
        destination: Path,
        *,
        expected_sha256: str | None = None,
        min_bytes: int | None = None,
        extra_headers: dict[str, str] | None = None,
        progress_label: str = "download",
        force: bool = False,
    ) -> Path:
        """Download ``url`` to ``destination`` safely.

        Returns the final path. Skips existing valid files unless ``force``
        is True (used when the operator explicitly requests a fresh copy).
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        min_bytes = min_bytes or self.settings.PIPELINE_MIN_VALID_BYTES

        if destination.exists():
            if self._existing_is_valid(destination, expected_sha256, min_bytes) and not force:
                log.info("Skipping %s: already exists and is valid.", destination.name)
                raise ExistingFileSkipped(str(destination))
            if self.settings.PIPELINE_SKIP_EXISTING and expected_sha256 is None and not force:
                log.info("Skipping %s: exists and no hash given for verification.", destination.name)
                raise ExistingFileSkipped(str(destination))
            if force:
                log.info("Forced refresh: removing existing %s before redownload.", destination.name)
            else:
                log.warning(
                    "Destroying existing %s (invalid hash or too small).", destination.name
                )
            destination.unlink()

        headers = {"User-Agent": "antarctic-navigation-dss/0.1.0"}
        if extra_headers:
            headers.update(extra_headers)

        retries = self.settings.PIPELINE_RETRIES
        backoff = self.settings.PIPELINE_RETRY_BACKOFF_SECONDS
        part = destination.with_suffix(destination.suffix + ".part")

        for attempt in range(1, retries + 1):
            log.info(
                "%s: attempt %d/%d -> %s", progress_label, attempt, retries, url
            )
            try:
                self._stream_download(url, headers, part, progress_label)
                if expected_sha256 and _sha256_of(part) != expected_sha256:
                    raise DownloadError(
                        f"SHA-256 mismatch for {destination.name}: expected "
                        f"{expected_sha256}"
                    )
                if part.stat().st_size < min_bytes:
                    raise DownloadError(
                        f"Downloaded file too small ({part.stat().st_size} bytes) "
                        f"for {destination.name}"
                    )
                # Atomic-ish rename now that the file is complete.
                part.replace(destination)
                log.info("%s: complete -> %s (%d bytes)", progress_label, destination, destination.stat().st_size)
                return destination
            except (urllib.error.URLError, DownloadError, OSError, TimeoutError) as exc:
                log.error("%s: attempt %d failed: %s", progress_label, attempt, exc)
                if attempt < retries:
                    log.info("Backing off %.1fs before retry.", backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    if part.exists():
                        part.unlink()
            except ExistingFileSkipped:
                raise

        raise DownloadError(
            f"Giving up on {url} after {retries} attempts (last target: {destination})"
        )

    def _existing_is_valid(
        self, path: Path, expected_sha256: str | None, min_bytes: int
    ) -> bool:
        if path.stat().st_size < min_bytes:
            return False
        if expected_sha256 and _sha256_of(path) != expected_sha256:
            return False
        return True

    def _stream_download(
        self, url: str, headers: dict[str, str], part: Path, label: str
    ) -> None:
        req = urllib.request.Request(url, headers=headers)
        timeout = self.settings.PIPELINE_TIMEOUT_SECONDS
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            total = int(content_length) if content_length else None

            written = 0
            with open(part, "wb") as out:
                while True:
                    chunk = resp.read(self.settings.PIPELINE_CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    self._log_progress(label, written, total)

            if total is not None and written != total:
                raise DownloadError(
                    f"Incomplete download: expected {total} bytes, wrote {written} "
                    f"(file {part.name} may be truncated)"
                )

    @staticmethod
    def _log_progress(label: str, written: int, total: int | None) -> None:
        if total:
            pct = 100.0 * written / total
            log.info("%s: %d / %d bytes (%.1f%%)", label, written, total, pct)
        else:
            log.info("%s: %d bytes", label, written)


def copy_local(src: Path, dst: Path) -> Path:
    """Copy a local source file into the raw store (used for NSIDC mirrors)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst