export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  demo_mode: boolean;
  database: string;
  timestamp: string;
  sea_ice_data: boolean;
  ocean_data: boolean;
  weather_data: boolean;
  iceberg_data: boolean;
  land_mask: boolean;
  iceberg_model: boolean;
  route_engine: boolean;
}

export interface DatasetInfo {
  dataset: string;
  description: string | null;
  classification: string;
  demo: boolean;
  status: string;
  file_path: string | null;
  records: number | null;
  source_id: string | null;
  source_name: string | null;
  notes: string | null;
  updated_at: string | null;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
  demo_mode: boolean;
  warning: string | null;
}

export interface DatasetStatusResponse {
  datasets: DatasetInfo[];
  real_data_available: boolean;
  demo_mode: boolean;
  warning: string | null;
}

export interface SeaIceCurrentResponse {
  timestamp: string;
  lat: number[];
  lon: number[];
  concentration: number[][];
  classification: string;
  demo: boolean;
  source: string | null;
  warning: string | null;
}

export interface SeaIceForecastResponse {
  forecast_time: string;
  valid_time: string | null;
  horizon_hours: number;
  model: string;
  lat: number[];
  lon: number[];
  concentration: number[][];
  mean_concentration: number;
  max_concentration: number;
  coverage_pct: number;
  classification: string;
  demo: boolean;
  model_used_real: boolean;
  skill_note: string | null;
  warning: string | null;
}

export interface IcebergInfo {
  iceberg_id: string;
  latitude: number;
  longitude: number;
  length_km: number | null;
  width_km: number | null;
  last_observed: string | null;
  source: string | null;
  demo: boolean;
  onboard_land?: boolean | null;
  on_land?: boolean | null;
  distance_to_coast_km?: number | null;
}

export interface IcebergsListResponse {
  icebergs: IcebergInfo[];
  count: number;
  classification: string;
  demo: boolean;
  warning: string | null;
}

export interface IcebergDetailResponse extends IcebergInfo {
  observation_count: number;
  classification: string;
  warning: string | null;
}

export interface IcebergPrediction {
  iceberg_id: string;
  current_lat: number;
  current_lon: number;
  predicted_lat: number;
  predicted_lon: number;
  prediction_hours: number;
  confidence: number | string | null;
  demo: boolean;
}

export interface IcebergPredictionsResponse {
  icebergs: IcebergPrediction[];
  model: string;
  horizon_hours: number;
  classification: string;
  demo: boolean;
  warning: string | null;
}

export interface TrajectoryPoint {
  timestamp: string | null;
  step: string;
  latitude: number;
  longitude: number;
  horizon_hours: number | null;
}

export interface IcebergTrajectoryResponse {
  iceberg_id: string;
  observations: TrajectoryPoint[];
  predictions: TrajectoryPoint[];
  count_observations: number;
  count_predictions: number;
  classification: string;
  demo: boolean;
  warning: string | null;
}

export interface IcebergDistanceResponse {
  iceberg_a: string;
  iceberg_b: string;
  distance_km: number;
  distance_nm: number;
  classification: string;
  demo: boolean;
  warning: string | null;
}

export interface RouteWaypoint {
  lat: number;
  lon: number;
  risk_score: number;
  step: number;
  [key: string]: unknown;
}

export interface RouteResult {
  waypoints: RouteWaypoint[];
  coordinates: [number, number][];
  distance_km: number;
  distance_nm: number;
  travel_time_hours: number;
  fuel_tons: number | null;
  risk_score: number | null;
  risk_level: string | null;
  warnings: string[];
}

export interface SeaIceHazardConfig {
  block_threshold: number;
  high_risk_threshold: number;
  moderate_risk_threshold: number;
  max_safe_concentration: number;
  vessel_max_concentration: number | null;
}

export interface IcebergHazardConfig {
  block_radius_km: number;
  warning_radius_km: number;
  prediction_horizon_hours: number;
  tracked_icebergs: number;
}

export interface HazardConfig {
  sea_ice: SeaIceHazardConfig;
  iceberg: IcebergHazardConfig;
  risk_weights: Record<string, number>;
  objective_weights: Record<string, number>;
  max_risk_ratio: number;
  hard_iceberg_block_applied: boolean;
}

export interface RouteGridInfo {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  resolution_degrees: number;
  nlat: number;
  nlon: number;
}

export interface RouteDebugInfo {
  grid: RouteGridInfo;
  land_cells: number;
  blocked_cells: number;
  forecast_horizon_hours: number;
  risk_weights: Record<string, number>;
  engine_notes: string[];
}

export interface RouteHazardConfig {
  sea_ice: {
    block_threshold: number;
    high_risk_threshold: number;
    moderate_risk_threshold: number;
    max_safe_concentration: number;
    vessel_max_concentration: number | null;
  };
  iceberg: {
    block_radius_km: number;
    warning_radius_km: number;
    prediction_horizon_hours: number;
    tracked_icebergs: number;
  };
  risk_weights: Record<string, number>;
  objective_weights: Record<string, number>;
  max_risk_ratio: number;
  hard_iceberg_block_applied: boolean;
}

export interface RouteDebugInfo {
  grid: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
    resolution_degrees: number;
    nlat: number;
    nlon: number;
  };
  land_cells: number;
  blocked_cells: number;
  forecast_horizon_hours: number;
  risk_weights: Record<string, number>;
  engine_notes: string[];
}

export interface RoutesOptimizeResponse {
  route_id: string;
  start_latitude: number;
  start_longitude: number;
  destination_latitude: number;
  destination_longitude: number;
  requested_start_latitude?: number | null;
  requested_start_longitude?: number | null;
  requested_destination_latitude?: number | null;
  requested_destination_longitude?: number | null;
  destination_name?: string | null;
  ocean_approach_distance_km?: number | null;
  vessel_id: string;
  preference: string;
  recommended: RouteResult;
  alternatives: RouteResult[];
  demo: boolean;
  disclaimer: string | null;
  hazard_config?: RouteHazardConfig | null;
  explanation?: string[];
  debug?: RouteDebugInfo | null;
}

export type AlertType = "route" | "sea-ice" | "iceberg" | "weather" | "system";
export type AlertSeverity = "info" | "low" | "medium" | "warning" | "high" | "critical";

export interface AlertRecord {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  routeId?: string;
  previousRoute?: RoutesOptimizeResponse | null;
  newRoute?: RoutesOptimizeResponse | null;
  reason?: string;
  latitude?: number;
  longitude?: number;
  icebergId?: string;
  metadata?: Record<string, string | number | null>;
  distanceKm?: number;
  distanceNm?: number;
  recommendedAction?: string;
  selectedRouteLabel?: string;
}

export interface RouteDetailResponse extends RoutesOptimizeResponse {
  created_at: string;
  waypoints_total: number;
}

export interface VesselInfo {
  vessel_id: string;
  name: string;
  vessel_type: string;
  ice_class: string | null;
  cruise_speed_knots: number | null;
  max_speed_knots: number | null;
  fuel_rate_lph: number | null;
  fuel_capacity_tons: number | null;
  safety_distance_nm: number | null;
}

export interface VesselsResponse {
  vessels: VesselInfo[];
  count: number;
}

export interface ModelAccuracyEntry {
  available: boolean;
  status: string | null;
  model?: string | null;
  reason?: string;
  n_test_pairs?: number;
  target?: string;
  unit?: string;
  horizon_hours?: number;
  mae?: number;
  rmse?: number;
  r2?: number;
  vs_zero_baseline?: boolean | null;
  mean_km?: number;
  median_km?: number;
  within_10km_pct?: number;
}

export interface EvaluationCheck {
  status: string;
  message?: string | null;
  target_percent?: number | null;
  accuracy_percent?: number | null;
  number_of_2h_prediction_pairs?: number | null;
  number_of_valid_2h_evaluations?: number | null;
  average_distance_error_km?: number | null;
  average_distance_error_nm?: number | null;
  missing_or_invalid_records?: number | null;
  total_predicted_points?: number | null;
  ocean_points?: number | null;
  land_points?: number | null;
  invalid_coordinates?: number | null;
}

export interface AccuracyBreakdownRow {
  component: string;
  label: string;
  accuracy: number | null;
  weight: number;
  weighted_contribution: number | null;
  status: string;
}

export interface AccuracySummary {
  score: number | null;
  provisional: boolean;
  available_weighted_sum: number;
  available_weight: number;
  available_weight_coverage_percent: number;
  missing_components: string[];
  status: string;
  grade: string;
  breakdown: AccuracyBreakdownRow[];
}

export interface ModelStatusEntry {
  model_name: string;
  display_name: string;
  deployed_model: string;
  status: string;
  training_data_source: string | null;
  metrics_available: boolean;
  metrics: Record<string, number>;
  trained_artifacts: { name: string; trained_on_demo: boolean | null }[];
}

export interface AnalyticsSummaryResponse {
  demo_mode: boolean;
  real_model_available: Record<string, boolean>;
  model_status: Record<string, ModelStatusEntry>;
  model_accuracy: Record<string, ModelAccuracyEntry>;
  evaluation_checks: Record<string, EvaluationCheck>;
  accuracy_summary: AccuracySummary;
  real_data_available: boolean;
  datasets: DatasetInfo[];
  routes_stored: number;
  icebergs_tracked: number;
  sea_ice_forecasts_stored: number;
  metrics_available: boolean;
  model_metrics_count: number;
  warnings: string[];
}

export interface TrainedModelInfo {
  pipeline: string;
  name: string;
  available: boolean;
  trained_on_demo: boolean | null;
  created_utc: string | null;
  artifact: string | null;
  error: string | null;
}

export interface ModelRegistryResponse {
  models: TrainedModelInfo[];
  count: number;
  any_real_trained: boolean;
  config: Record<string, string>;
  demo_mode: boolean;
  warning: string | null;
}

export type PageId =
  | "home"
  | "sea-ice"
  | "icebergs"
  | "planner"
  | "alerts"
  | "assistant";

export interface AssistantMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AssistantChatRequest {
  question: string;
  history: AssistantMessage[];
  horizon_hours?: number;
}

export interface AssistantContextSource {
  name: string;
  demo: boolean;
  note: string | null;
  status: string;
}

export interface AssistantChatResponse {
  answer: string;
  sources: AssistantContextSource[];
  llm: {
    provider: string;
    model: string | null;
    configured: boolean;
  };
  demo_mode: boolean;
  warnings: string[];
}

export interface PortInfo {
  port_id: string;
  name: string;
  country: string;
  latitude: number;
  longitude: number;
  description: string;
  facilities: string[];
  timezone: string;
}

export interface ResearchCenterInfo {
  center_id: string;
  name: string;
  latitude: number;
  longitude: number;
  country: string;
  description: string;
  sector: string;
  active: boolean;
}

export interface LegacyVesselInfo {
  vessel_id: string;
  name: string;
  vessel_type: string;
  ice_class: string;
  max_speed_knots: number;
  cruise_speed_knots: number;
  fuel_rate_lph: number;
  safety_distance_nm: number;
  fuel_capacity_tons: number;
  fuel_consumption_tons_per_day: Record<string, number>;
  range_nautical_miles: Record<string, number>;
}

export interface SimulationConfig {
  time_step_hours: number;
  [key: string]: unknown;
}

export type JourneyMode = "outbound" | "return";

export interface JourneyWaypoint {
  latitude: number;
  longitude: number;
  distance_nm: number;
  heading_deg: number;
  speed_knots: number;
  sea_ice_concentration: number;
  iceberg_risk: number;
  risk_level: string;
  estimated_fuel_tons: number;
  timestamp_hours: number;
}

export interface JourneyResponse {
  journey_id: string;
  origin: string;
  destination: string;
  journey_mode: JourneyMode;
  waypoints: JourneyWaypoint[];
  total_distance_nm: number;
  total_fuel_tons: number;
  estimated_duration_hours: number;
  max_risk_level: string;
  route_update_count: number;
  current_vessel_lat: number;
  current_vessel_lon: number;
  created_at: string;
}

export interface JourneyStatus {
  journey_id: string;
  current_time_hours: number;
  current_lat: number;
  current_lon: number;
  progress_percent: number;
  remaining_distance_nm: number;
  remaining_fuel_tons: number;
  is_complete: boolean;
  route_update_count: number;
}

export interface StartJourneyRequest {
  departure_port_id: string;
  destination_id: string;
  vessel_id?: string;
  journey_mode?: JourneyMode;
  position_mode?: "simulation" | "live";
  current_latitude?: number;
  current_longitude?: number;
}