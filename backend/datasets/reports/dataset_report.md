# Dataset Validation Report - 20260911T095644Z

- Generated: 2026-09-11T09:56:44.710385+00:00
- Overall status: **valid**

## Credential configuration

- NSIDC: not configured
- Copernicus Marine: not configured
- CDS/ERA5: not configured

## Datasets

### icebergs
- `icebergs_demo.csv` - **VALID**

### ocean
- `ocean_surface.nc` - **VALID**
  - [warning] Synthetic demo data; not real measurements.

### sea_ice
- `sea_ice.nc` - **VALID**
  - [warning] File contains synthetic demo data and must not be used operationally.

### weather
- `weather_surface.nc` - **VALID**
  - [warning] Synthetic demo data; not real measurements.

## Data freshness (two-hour update support)

### icebergs_demo
- **file**: icebergs_demo.nc
- **classification**: synthetic_demo
- **source_name**: synthetic-demo-generator
- **cadence_hours**: 6.0
- **note**: Data cadence is 6.0 hourly; navigation queries use the recorded timestamps and are interpolated only when scientifically appropriate. Demo data is not real.

### icebergs_sample
- **file**: icebergs_sample.nc
- **classification**: imported
- **source_name**: import
- **cadence_hours**: 12.0
- **note**: Data cadence is 12.0 hourly; navigation queries use the recorded timestamps and are interpolated only when scientifically appropriate. Demo data is not real.

### ocean_surface
- **file**: ocean_surface.nc
- **classification**: synthetic_demo
- **source_name**: synthetic-demo-generator
- **cadence_hours**: 24.0
- **note**: Data cadence is 24.0 hourly; navigation queries use the recorded timestamps and are interpolated only when scientifically appropriate. Demo data is not real.

### sea_ice
- **file**: sea_ice.nc
- **classification**: synthetic_demo
- **source_name**: synthetic-demo-generator
- **cadence_hours**: 24.0
- **note**: Data cadence is 24.0 hourly; navigation queries use the recorded timestamps and are interpolated only when scientifically appropriate. Demo data is not real.

### weather_surface
- **file**: weather_surface.nc
- **classification**: synthetic_demo
- **source_name**: synthetic-demo-generator
- **cadence_hours**: 6.0
- **note**: Data cadence is 6.0 hourly; navigation queries use the recorded timestamps and are interpolated only when scientifically appropriate. Demo data is not real.

---
Demo/labeled synthetic data is NOT real and must never be used for operational navigation decisions.
