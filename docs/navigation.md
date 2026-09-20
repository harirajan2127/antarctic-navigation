# Navigation

## Grid & risk engine (`backend/navigation/`)

- **Domain**: latitude −85°…−55°, longitude −180°…180°, resolution 0.5°.
- **Projection**: equirectangular, `AntarcticGrid`.
- **Land mask**: pre-aligned Antarctic land mask loaded from numpy (`.npy`)
  cells; land cells are excluded from navigation.
- **Risk layers** (`RiskEngine` + `RiskConfig`):
  - sea-ice concentration mapped to a risk weight (vessel-ice-class limits,
    e.g. PC5 → 60 % sea-ice ceiling),
  - hard iceberg No-Go blocks — cells whose closest approach to a known
    iceberg is below a threshold are impassable obstacles,
  - weather severity,
  - vessel constraint layers.
- **A\*** path planning (`AStarPlanner`) over the risk-weighted cost field with
  obstacle avoidance + a `NoPathFoundError` for impossible journeys.
- **Routing** (`RouteOptimizer`): shortest / safest / fuel-efficient /
  recommended preferences; distances from summed haversine between waypoints.
  `RouteValidator` checks the final route.
- **Fuel/time** (`FuelEstimator`): from the vessel record (ice class, speed,
  fuel model) — no hard-coded rates.

Thresholds and objective weights are in
`backend/app/data/config/navigation.json` and
`backend/data/config/simulation.json`.

## Journey lifecycle (legacy `/api/v1`)

`POST /api/v1/navigation/journey`

```json
{
  "departure_port_id": "hobart_au",
  "destination_id": "casey_au",
  "journey_mode": "outbound",
  "position_mode": "simulation"
}
```

- `journey_mode`: `outbound` (port → center) or `return` (center → port).
- `position_mode`: `simulation` (advance along route) or `live` (requires
  `POST .../recalculate?lat=&lon=` GPS input; returns 400 otherwise).
- Response: origin/destination names, current position, total distance (nm),
  estimated duration (h) and fuel (t), `max_risk_level`, waypoints, and a
  `route_update_count`.

`POST /api/v1/navigation/journey/{id}/advance` — advances two simulated hours,
re-optimizes the route **from the current vessel position**, keeps the
destination fixed, and increments the update counter. Position is never reset
to the departure port.

`POST /api/v1/navigation/journey/{id}/recalculate?lat=&lon=` — re-plans from an
external (e.g. GPS) fix.

`GET /api/v1/navigation/journey/{id}/status` — current position, elapsed
`current_time_hours`, `remaining_distance_nm`, `progress_percent`, fuel,
`is_complete`.

## Known grid behaviors (see `docs/limitations.md`)

1. **Northern-port clamping** — ports north of the grid (Hobart −42.9°,
   Cape Town −33.9°, Ushuaia −54.8°) cannot lie on the grid, so the planned
   route starts at the grid edge (−55°). Reported `total_distance_nm` is the
   in-grid route length and is **less** than the true great-circle distance
   (e.g. Hobart→McMurdo API ≈ 1483 nm vs great-circle ≈ 2151 nm).
2. **Destination cell-center snap** — the destination resolves to a 0.5° cell
   center, so an arriving vessel's remaining distance stays bounded (< 30 nm)
   while `is_complete=true`. Recalculating exactly at the destination yields a
   single-waypoint (zero-distance) route whose `progress_percent` reads 0 even
   though the journey is complete.