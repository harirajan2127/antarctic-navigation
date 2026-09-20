# Antarctic DSS - Real Model Accuracy Evaluation Report

- **Evaluation date/time:** 2026-09-17T17:19:26+00:00
- **Project root:** `C:\my project\final\SIH HACK`
- **Data mode:** real (`Real data/processed`)

## 1. Evaluation Summary Status

| Section | Status |
|---------|--------|
| data_verification | PASSED |
| sea_ice_model | VERIFIED |
| iceberg_model | VERIFIED |
| single_route | VERIFIED |
| two_hour_recalc | INSUFFICIENT_DATA |
| land_mask | VERIFIED |
| route_engine | VERIFIED |

## 2. Datasets Verified

### iceberg
- File: `iceberg_processed.csv`
- Rows: 2963  Columns: ['iceberg_id', 'timestamp', 'latitude', 'longitude', 'length_nm', 'width_nm', 'remarks', 'source_file', 'source_format']
- Missing: {'iceberg_id': 0, 'timestamp': 0, 'latitude': 0, 'longitude': 0, 'length_nm': 0, 'width_nm': 0, 'remarks': 720, 'source_file': 0, 'source_format': 0}
- Duplicate rows: 0
- latitude_range: [-74.97, -55.0]
- longitude_range: [-167.67, 148.14]
- timestamp_range: ['2022-01-07T00:00:00+00:00', '2023-12-21T00:00:00+00:00']
- distinct_icebergs: 65
- ground_truth_future: True

### sea_ice
- File: `sea_ice_part_202312.csv`
- Rows: 5000000  Columns: ['timestamp', 'latitude', 'longitude', 'sea_ice_concentration', 'nodata_flag']
- Missing: {'timestamp': 0, 'latitude': 0, 'longitude': 0, 'sea_ice_concentration': 2392521, 'nodata_flag': 0}
- Duplicate rows: 0
- latitude_range: [-75.0, -55.0]
- longitude_range: [-180.0, 179.75]
- timestamp_range: ['2023-02-02T00:00:00+00:00', '2023-12-31T00:00:00+00:00']
- distinct_timestamps: 55
- concentration_range: [0.0, 1.0]
- concentration_unit: fraction (0..1)

## 3. Sea-Ice Model Accuracy (deployed persistence model)

- **Model:** persistence (no learned parameters; deployed model)
- **Target:** sea_ice_concentration (fraction 0..1) (fraction)
- **Evaluation file:** sea_ice_part_202312.csv
- **Test horizon (mean gap):** 24.0 h
- **Split (chronological):** 38 train-ref dates, 5 val dates, 12 test dates
- **Test record count (cell-day pairs):** 1013895
- Test metrics: MAE=0.0360, RMSE=0.1088, MSE=0.0118, R2=0.8981, explainedVar=0.8983, bias=-0.0046, corr=0.9498
- Invalid predictions (outside 0..1): 0

#### Error by latitude band (test)

- lat -75--70: MAE=0.0787 RMSE=0.1597 n=135799
- lat -70--65: MAE=0.0798 RMSE=0.1602 n=237745
- lat -65--60: MAE=0.0220 RMSE=0.0879 n=309382
- lat -60--55: MAE=0.0002 RMSE=0.0099 n=315354

#### Error by observed concentration range (test)

- conc 0-0.2: MAE=0.0115 RMSE=0.0684 n=791053
- conc 0.2-0.4: MAE=0.2359 RMSE=0.2875 n=24431
- conc 0.4-0.6: MAE=0.2211 RMSE=0.2638 n=27514
- conc 0.6-0.8: MAE=0.1801 RMSE=0.2327 n=37877
- conc 0.8-1.0: MAE=0.0808 RMSE=0.1455 n=102015

#### Baseline comparison

- Persistence (deployed model): same numbers above (model == baseline).
- Zero-prediction baseline on test: MAE=0.1745, RMSE=0.3830
- Persistence beats zero-baseline: True

## 4. Iceberg Trajectory Model Accuracy (deployed model)

- **Model:** persistence (last known position)
- **Test pairs (real consecutive obs):** 415
- **Test icebergs:** 61
- **Latitude MAE:** 0.0511 deg
- **Longitude MAE:** 0.2628 deg
- **Mean Haversine error:** 16.42 km
- **Median Haversine error:** 0.91 km
- **Max Haversine error:** 1831.31 km
- **Position RMSE:** 130.61 km
- Within-threshold percentages:
- By horizon bucket (real gaps):
- Note: Random forest trained on cleaned real iceberg observations with a chronological hold-out. No future observations were used as features.

## 5. Route Engine Evaluation (real data, production engine)

- **Status:** VERIFIED  classification=pipeline_data
- **Distance:** 3792.7 km / 2047.9 nm
- **Travel time:** 170.7 h
- **Waypoints:** 80  **Fuel:** 199.1 t
- **Risk score:** 0.1217 level=moderate
- **Generation time:** 14.94 s
- **Starts at departure:** True
- **Ends at destination:** True
- WARN: Iceberg predictions use trained model 'random_forest' on real observations.
- WARN: Weather severity derived from the newest real wind field.
- WARN: Land mask loaded (json): 23967 cells marked land.
- WARN: Iceberg tracking hazard within 60 km at (-66.00, 110.00) (43.9 km).
- WARN: Iceberg tracking hazard within 60 km at (-66.00, 110.50) (48.5 km).
- WARN: Iceberg tracking hazard within 60 km at (-66.50, 110.50) (23.6 km).

## 6. Two-Hour Rolling Recalculation

- **Status:** INSUFFICIENT_DATA
- **Message:** Insufficient 2-hour ground-truth data
- **Dataset interval:** 168.0 hours
- **Iceberg tracks:** 65
- **2-hour prediction pairs:** 0
- **Valid evaluations:** 0
- **Average error:** None km / None nm
- **Accuracy score:** None% (tolerance=20.0 km)
- **Missing or invalid records:** 44

## 7. Land Mask

- **Status:** VERIFIED
- **Format:** json
- **Land cells (on grid):** 23967  fraction=0.5449
- **Predicted points:** 65
- **Ocean points:** 48
- **Land points:** 17
- **Invalid coordinates:** 0
- **Ocean-valid score:** 73.8462%

## 8. Data Leakage Checks

- chronological_split_used: Yes — dates/observations ordered in time; test uses only the latest window.
- future_values_used_as_inputs: No
- train_test_overlap: No — distinct dates/windows.
- evaluated_on_training_records: No

## 9. Missing Data / Limitations

- The 2-hour score uses only consecutive real observations within the configured 2-hour interval tolerance; interpolated points are never counted as ground truth.
- Sea-ice persistence metrics were computed from a single monthly part of the real CSV set.
- Iceberg observations are ~weekly, so sub-week horizons without direct ground truth are reported as insufficient rather than scored.
- This is software/research validation, not a real-world navigation safety claim.

## 10. Final Evaluation Status

| Item | Status |
|------|--------|
| Sea-ice model accuracy | VERIFIED |
| Iceberg model accuracy | VERIFIED |
| Route-engine validation | VERIFIED |
| Two-hour recalc | INSUFFICIENT_DATA |
| Land mask active | VERIFIED |

> This report is for software and research validation only. It is NOT a
> claim that the system is safe for real-world navigation.
