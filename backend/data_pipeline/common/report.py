"""Dataset report generation.

Produces a structured report (JSON + Markdown) describing every configured
dataset: availability, validation outcome, freshness, and interpolation
metadata. The report is the single source of truth for the dashboard and
for operators.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.validation import ValidationResult

log = get_logger(__name__)


class DatasetReport:
    """Accumulates per-dataset validation results and writes the report."""

    def __init__(self) -> None:
        self.created_at = datetime.now(timezone.utc)
        self.datasets: dict[str, list[ValidationResult]] = {}
        self.freshness: dict[str, dict[str, Any]] = {}

    def add_result(self, result: ValidationResult) -> None:
        self.datasets.setdefault(result.dataset, []).append(result)

    def add_freshness(self, dataset: str, info: dict[str, Any]) -> None:
        self.freshness[dataset] = info

    def overall_status(self) -> str:
        if not self.datasets:
            return "empty"
        worst = "valid"
        for results in self.datasets.values():
            for r in results:
                if r.status == "invalid":
                    worst = "invalid"
                elif r.status == "missing" and worst != "invalid":
                    worst = "missing"
                elif r.status == "incomplete" and worst in ("valid",):
                    worst = "incomplete"
        return worst

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.created_at.strftime("%Y%m%dT%H%M%SZ"),
            "generated_at": self.created_at.isoformat(),
            "overall_status": self.overall_status(),
            "credential_config": {
                "nsidc": bool(
                    get_pipeline_settings().has_nsidc_credentials
                ),
                "copernicus_marine": bool(
                    get_pipeline_settings().has_copernicus_credentials
                ),
                "cds_era5": bool(get_pipeline_settings().has_cds_credentials),
            },
            "datasets": {
                name: [r.to_dict() for r in results]
                for name, results in self.datasets.items()
            },
            "freshness": self.freshness,
            "disclaimer": (
                "Demo/labeled synthetic data is NOT real and must never be used for "
                "operational navigation decisions."
            ),
        }

    def write(self, report_dir: Path | None = None, name: str = "dataset_report") -> tuple[Path, Path]:
        """Write JSON + Markdown report files, returning both paths."""
        settings = get_pipeline_settings()
        report_dir = report_dir or settings.DATA_REPORT_ROOT
        report_dir.mkdir(parents=True, exist_ok=True)

        payload = self.to_dict()
        json_path = report_dir / f"{name}.json"
        md_path = report_dir / f"{name}.md"

        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self._to_markdown(payload), encoding="utf-8")
        log.info("Report written: %s", json_path)
        return json_path, md_path

    @staticmethod
    def _to_markdown(payload: dict[str, Any]) -> str:
        lines = [
            f"# Dataset Validation Report - {payload['report_id']}",
            "",
            f"- Generated: {payload['generated_at']}",
            f"- Overall status: **{payload['overall_status']}**",
            "",
            "## Credential configuration",
            "",
            f"- NSIDC: {'configured' if payload['credential_config']['nsidc'] else 'not configured'}",
            f"- Copernicus Marine: {'configured' if payload['credential_config']['copernicus_marine'] else 'not configured'}",
            f"- CDS/ERA5: {'configured' if payload['credential_config']['cds_era5'] else 'not configured'}",
            "",
            "## Datasets",
            "",
        ]
        for name, results in sorted(payload["datasets"].items()):
            lines.append(f"### {name}")
            for r in results:
                lines.append(f"- `{r['file']}` - **{r['status'].upper()}**")
                for c in r["checks"]:
                    lines.append(f"  - [{c['severity']}] {c['message']}")
            lines.append("")
        if payload.get("freshness"):
            lines.append("## Data freshness (two-hour update support)")
            lines.append("")
            for ds, info in payload["freshness"].items():
                lines.append(f"### {ds}")
                for k, v in info.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")
        lines.append("---")
        lines.append(payload["disclaimer"])
        lines.append("")
        return "\n".join(lines)