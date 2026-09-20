"""Dataset validation utilities.

Generic rules that the domain modules apply to their files. Produces a
machine-readable validation result per file plus aggregated report data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from data_pipeline.common.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    detail: Any = None


@dataclass
class ValidationResult:
    dataset: str
    file: str
    status: str  # "valid" | "incomplete" | "invalid" | "missing"
    checks: list[ValidationIssue] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.checks)

    def add(self, severity: str, code: str, message: str, detail: Any = None) -> None:
        self.checks.append(ValidationIssue(severity, code, message, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "file": self.file,
            "status": self.status,
            "checks": [
                {"severity": c.severity, "code": c.code, "message": c.message, "detail": c.detail}
                for c in self.checks
            ],
            "meta": self.meta,
        }


def validate_time_range(values: list[Any], start: datetime, end: datetime) -> list[ValidationIssue]:
    """Check that all timestamps fall within [start, end]."""
    issues: list[ValidationIssue] = []
    if not values:
        issues.append(ValidationIssue("error", "TIME_EMPTY", "No timestamps present"))
        return issues
    tmin = min(values)
    tmax = max(values)
    if tmin < start:
        issues.append(ValidationIssue("error", "TIME_BEFORE_RANGE", f"Earliest {tmin} < {start}"))
    if tmax > end:
        issues.append(ValidationIssue("error", "TIME_AFTER_RANGE", f"Latest {tmax} > {end}"))
    issues.append(ValidationIssue("info", "TIME_SPAN", f"{tmin} -> {tmax}", {"count": len(values)}))
    return issues


def validate_coordinate_range(
    name: str, values: list[float], lo: float, hi: float
) -> ValidationIssue | None:
    """Check a 1D coordinate stays within [lo, hi]."""
    if not values:
        return ValidationIssue("error", "COORD_EMPTY", f"Coordinate {name} empty")
    mn, mx = min(values), max(values)
    if mn < lo or mx > hi:
        return ValidationIssue(
            "error",
            "COORD_OUT_OF_RANGE",
            f"Coordinate {name} range [{mn}, {mx}] outside [{lo}, {hi}]",
            {"lo": mn, "hi": mx},
        )
    return None


def validate_missing_values(name: str, values: Any, max_fraction: float = 0.01) -> ValidationIssue | None:
    """Count NaN/None values and warn beyond ``max_fraction``."""
    import numpy as np

    arr = np.asarray(values)
    missing = int(np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0)
    total = arr.size
    if total == 0:
        return ValidationIssue("error", "FIELD_EMPTY", f"Field {name} has no data")
    frac = missing / total
    if frac > max_fraction:
        return ValidationIssue(
            "warning",
            "MISSING_VALUES",
            f"Field {name}: {missing}/{total} missing values ({frac:.2%} > {max_fraction:.0%})",
            {"missing": missing, "total": total},
        )
    return None


def check_field_units(name: str, unit_attr: str | None, expected: str) -> ValidationIssue | None:
    """Check that a variable's units attribute matches the expectation."""
    if unit_attr is None:
        return ValidationIssue(
            "warning", "UNITS_MISSING", f"Field {name} has no units attribute", expected
        )
    if unit_attr.strip().lower() != expected.lower():
        return ValidationIssue(
            "error",
            "UNITS_MISMATCH",
            f"Field {name} units '{unit_attr}' != expected '{expected}'",
            {"actual": unit_attr, "expected": expected},
        )
    return None


def numeric_range_check(
    name: str, values: Any, lo: float, hi: float
) -> ValidationIssue | None:
    """Check a numeric field stays within [lo, hi]."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return ValidationIssue("error", "FIELD_EMPTY", f"Field {name} empty")
    f = np.isfinite(arr)
    if not f.all():
        return ValidationIssue(
            "warning", "NON_FINITE", f"Field {name} contains non-finite values"
        )
    mn, mx = arr[f].min(), arr[f].max()
    if mn < lo or mx > hi:
        return ValidationIssue(
            "error",
            "FIELD_OUT_OF_RANGE",
            f"Field {name} range [{mn}, {mx}] outside [{lo}, {hi}]",
            {"lo": float(mn), "hi": float(mx)},
        )
    return None


def summarize_result(result: ValidationResult) -> str:
    """Render a one-line status summary for logging."""
    if result.status == "missing":
        return f"[{result.dataset}] {result.file}: MISSING"
    summary = (
        f"[{result.dataset}] {result.file}: {result.status.upper()} "
        f"({len(result.checks)} checks)"
    )
    return summary