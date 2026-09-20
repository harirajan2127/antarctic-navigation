# Limitations

Honest, current limitations of the system — read before operational use.

## Data & ML
- **Synthetic demo data only.** Offline workflows generate explicitly-labeled
  `synthetic_demo` fields; no real observations are being ingested, so
  `real_data_available=false`. No real navigation decisions should be made
  from demo data.
- **No real model skill claimed.** All shipped checkpoint metrics are
  code-path checks from `--demo` training; models are `trained_on_demo: true`.
  Iceberg demo tracks drift unrealistically; persistence often outperforms ML
  at multi-step horizons.
- **Observation cadence ≠ 2 h loop.** Remote fields are daily (sea-ice/ocean)
  or 6-hourly (ERA5) while navigation advances every 2 simulated hours. The
  accessor serves the closest valid observation with provenance — it never
  pretends daily data is 2-hourly.

## Navigation grid
- **Domain clamp for northern ports.** The risk grid covers
  latitude −85°…−55°. Ports north of −55° (Hobart −42.9°, Cape Town −33.9°,
  Ushuaia −54.8°) cannot lie on the grid, so a planned route starts at the
  grid edge (−55°). `total_distance_nm` is the **in-grid** route length and is
  less than the true great-circle distance (Hobart→McMurdo ≈ 1483 nm vs
  ≈ 2151 nm).
- **Destination cell-center snap.** A destination resolves to a 0.5° cell
  center; arriving vessels keep a bounded residual (< ~30 nm) with
  `is_complete=true`. Recalculating exactly at the destination produces a
  single-waypoint zero-distance route whose `progress_percent` reads 0 even
  though the journey is complete.
- **Static demo environment.** Demo sea-ice/iceberg fields do not change
  between 2-hour steps, so in demo mode the "re-route due to environmental
  change" behavior is exercised as recalculation-from-current-position rather
  than as a response to a changing risk field. Live fields would re-route on
  change; this is untested against real data.

## Live position
- No actual GPS feed is implemented. `position_mode: "live"` returns 400 until
  a real fix is supplied via `POST .../recalculate?lat=&lon=`. Simulation mode
  advances along the planned route.

## Infrastructure
- Local single-machine deployment: SQLite (SQLAlchemy), in-memory journey
  state (journeys are lost on restart), no PostGIS, no auth, no horizontal
  scaling.
- The unified backend (`:8000`) and legacy journey API (`:8001`) share
  services but are separate uvicorn processes; the resource-heavy route
  planners (A\* over 61 × 721 cells) run synchronously.
- Windows-only bootstrap scripts (`run_demo.ps1`).

## Safety
- Routes and risk scores are **decision support only — never guaranteed safe**.
  Not a certified navigation system; operational decisions require qualified
  crew, official charts and ice briefings.