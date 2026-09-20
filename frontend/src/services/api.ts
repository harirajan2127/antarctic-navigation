import axios from "axios";
import type {
  AnalyticsSummaryResponse,
  AssistantChatRequest,
  AssistantChatResponse,
  DatasetStatusResponse,
  DatasetsResponse,
  HealthResponse,
  IcebergDetailResponse,
  IcebergDistanceResponse,
  IcebergPredictionsResponse,
  IcebergTrajectoryResponse,
  IcebergsListResponse,
  JourneyResponse,
  JourneyStatus,
  LegacyVesselInfo,
  ModelRegistryResponse,
  PortInfo,
  ResearchCenterInfo,
  RouteDetailResponse,
  RoutesOptimizeResponse,
  SeaIceCurrentResponse,
  SeaIceForecastResponse,
  SimulationConfig,
  StartJourneyRequest,
  VesselsResponse,
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const http = axios.create({
  baseURL: `${API_BASE_URL ?? ""}/api`,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

const routeHttp = axios.create({
  baseURL: `${API_BASE_URL ?? ""}/api`,
  timeout: 120000,
  headers: { "Content-Type": "application/json" },
});

routeHttp.interceptors.response.use(
  (r) => r,
  (err) => {
    const msg = err.response?.data?.detail ?? err.message ?? "Route request failed";
    return Promise.reject(new Error(typeof msg === "string" ? msg : JSON.stringify(msg)));
  },
);

http.interceptors.response.use(
  (r) => r,
  (err) => {
    const msg =
      err.response?.data?.detail ??
      err.response?.data ??
      err.message ??
      "Unknown error";
    return Promise.reject(new Error(typeof msg === "string" ? msg : JSON.stringify(msg)));
  },
);

export const api = {
  health: () =>
    http.get<HealthResponse>("/health").then((r) => r.data),

  datasets: () =>
    http.get<DatasetsResponse>("/datasets").then((r) => r.data),

  datasetsStatus: () =>
    http.get<DatasetStatusResponse>("/datasets/status").then((r) => r.data),

  seaIceCurrent: () =>
    http.get<SeaIceCurrentResponse>("/sea-ice/current").then((r) => r.data),

  seaIceForecast: (horizonHours: number, model = "persistence") =>
    http
      .get<SeaIceForecastResponse>("/sea-ice/forecast", {
        params: { horizon_hours: horizonHours, model },
      })
      .then((r) => r.data),

  icebergs: () =>
    http.get<IcebergsListResponse>("/icebergs").then((r) => r.data),

  icebergDetail: (id: string) =>
    http.get<IcebergDetailResponse>(`/icebergs/${id}`).then((r) => r.data),

  icebergPredict: (icebergIds: string[], horizonHours = 24) =>
    http
      .post<IcebergPredictionsResponse>("/icebergs/predict", {
        iceberg_ids: icebergIds,
        horizon_hours: horizonHours,
      })
      .then((r) => r.data),

  icebergTrajectory: (id: string) =>
    http
      .get<IcebergTrajectoryResponse>(`/icebergs/${id}/trajectory`)
      .then((r) => r.data),

  icebergDistance: (a: string, b: string) =>
    http
      .get<IcebergDistanceResponse>("/icebergs/distance", {
        params: { iceberg_a: a, iceberg_b: b },
      })
      .then((r) => r.data),

  routesOptimize: (payload: {
    start_latitude: number;
    start_longitude: number;
    destination_latitude: number;
    destination_longitude: number;
    vessel_id: string;
    preference: string;
  }) =>
    routeHttp.post<RoutesOptimizeResponse>("/routes/optimize", payload).then((r) => r.data),

  routeDetail: (id: string) =>
    http.get<RouteDetailResponse>(`/routes/${id}`).then((r) => r.data),

  vessels: () =>
    http.get<VesselsResponse>("/vessels").then((r) => r.data),

  // ---- Legacy /api/v1 config + journey lifecycle (mounted on the same
  // backend, so it flows through the same Vite proxy) ----
  ports: () =>
    http.get<PortInfo[]>("/v1/config/ports").then((r) => r.data),

  researchCenters: () =>
    http.get<ResearchCenterInfo[]>("/v1/config/research-centers").then((r) => r.data),

  legacyVessels: () =>
    http.get<LegacyVesselInfo[]>("/v1/vessels/").then((r) => r.data),

  simulationConfig: () =>
    http.get<SimulationConfig>("/v1/config/simulation").then((r) => r.data),

  startJourney: (payload: StartJourneyRequest) =>
    http.post<JourneyResponse>("/v1/navigation/journey", payload).then((r) => r.data),

  advanceJourney: (journeyId: string) =>
    http.post<JourneyResponse>(`/v1/navigation/journey/${journeyId}/advance`).then((r) => r.data),

  recalculateJourney: (journeyId: string, lat: number, lon: number) =>
    http
      .post<JourneyResponse>(`/v1/navigation/journey/${journeyId}/recalculate`, null, {
        params: { lat, lon },
      })
      .then((r) => r.data),

  journeyStatus: (journeyId: string) =>
    http.get<JourneyStatus>(`/v1/navigation/journey/${journeyId}/status`).then((r) => r.data),

  analyticsSummary: () =>
    http.get<AnalyticsSummaryResponse>("/analytics/summary").then((r) => r.data),

  modelsStatus: () =>
    http.get<ModelRegistryResponse>("/models/status").then((r) => r.data),

  assistantChat: (payload: AssistantChatRequest) =>
    http.post<AssistantChatResponse>("/assistant/chat", payload).then((r) => r.data),
};