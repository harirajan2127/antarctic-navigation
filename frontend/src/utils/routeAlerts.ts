import type {
  AlertRecord,
  AlertSeverity,
  IcebergInfo,
  RouteHazardConfig,
  RouteResult,
  SeaIceForecastResponse,
} from "../types";
import { haversineKm } from "./haversine";

const KM_TO_NM = 0.5399568;

export interface RouteAlertInputs {
  route: RouteResult;
  routeId: string;
  routeLabel: string;
  seaIce: SeaIceForecastResponse | null;
  icebergs: IcebergInfo[];
  hazardConfig?: RouteHazardConfig | null;
}

export function distanceToRoute(
  point: [number, number],
  coordinates: [number, number][],
): number {
  if (!coordinates.length) return Number.POSITIVE_INFINITY;
  let minimum = Number.POSITIVE_INFINITY;
  coordinates.forEach((coordinate) => {
    minimum = Math.min(minimum, haversineKm(point[0], point[1], coordinate[0], coordinate[1]));
  });
  for (let index = 1; index < coordinates.length; index += 1) {
    const start = coordinates[index - 1];
    const end = coordinates[index];
    for (let step = 1; step < 5; step += 1) {
      const fraction = step / 5;
      const candidate: [number, number] = [
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
      ];
      minimum = Math.min(minimum, haversineKm(point[0], point[1], candidate[0], candidate[1]));
    }
  }
  return minimum;
}

function icebergSeverity(distanceKm: number, config?: RouteHazardConfig | null): AlertSeverity {
  const critical = config?.iceberg.block_radius_km ?? 5;
  const high = config?.iceberg.warning_radius_km ?? 15;
  if (distanceKm < critical) return "critical";
  if (distanceKm <= high) return "high";
  if (distanceKm <= 30) return "medium";
  return "low";
}

function seaIceSeverity(concentration: number, config?: RouteHazardConfig | null): AlertSeverity {
  const critical = config?.sea_ice.block_threshold ?? 0.8;
  const high = config?.sea_ice.high_risk_threshold ?? 0.6;
  const medium = config?.sea_ice.moderate_risk_threshold ?? 0.3;
  if (concentration >= critical) return "critical";
  if (concentration >= high) return "high";
  if (concentration >= medium) return "medium";
  return "low";
}

function alertBase(
  type: AlertRecord["type"],
  severity: AlertSeverity,
  title: string,
  message: string,
  input: RouteAlertInputs,
  latitude: number | undefined,
  longitude: number | undefined,
  distanceKm: number | undefined,
  recommendedAction: string,
): AlertRecord {
  return {
    id: `${type}-${input.routeId}-${input.routeLabel}-${latitude?.toFixed(4) ?? "unavailable"}-${longitude?.toFixed(4) ?? "unavailable"}`,
    type,
    severity,
    title,
    message,
    timestamp: new Date().toISOString(),
    read: false,
    routeId: input.routeId,
    selectedRouteLabel: input.routeLabel,
    latitude,
    longitude,
    distanceKm,
    distanceNm: distanceKm == null ? undefined : distanceKm * KM_TO_NM,
    recommendedAction,
  };
}

export function calculateRouteAlerts(input: RouteAlertInputs): AlertRecord[] {
  const alerts: AlertRecord[] = [];
  const coordinates = input.route.coordinates;

  if (input.seaIce?.lat.length && input.seaIce.lon.length && input.seaIce.concentration.length) {
    const cells: { distanceKm: number; latitude: number; longitude: number; concentration: number }[] = [];
    input.seaIce.concentration.forEach((row, rowIndex) => row.forEach((concentration, columnIndex) => {
      const latitude = input.seaIce?.lat[rowIndex];
      const longitude = input.seaIce?.lon[columnIndex];
      if (latitude == null || longitude == null || !Number.isFinite(concentration)) return;
      const distanceKm = distanceToRoute([latitude, longitude], coordinates);
      cells.push({ distanceKm, latitude, longitude, concentration });
    }));
    const nearestPoint = cells.reduce<typeof cells[number] | null>(
      (current, cell) => !current || cell.distanceKm < current.distanceKm ? cell : current,
      null,
    );
    if (nearestPoint) {
      const severity = seaIceSeverity(nearestPoint.concentration, input.hazardConfig);
      alerts.push({
        ...alertBase(
          "sea-ice",
          severity,
          "Nearest Sea-Ice Alert",
          `${severity.toUpperCase()}: Sea-ice concentration of ${(nearestPoint.concentration * 100).toFixed(0)}% detected ${nearestPoint.distanceKm.toFixed(1)} km from the ${input.routeLabel} route.`,
          input,
          nearestPoint.latitude,
          nearestPoint.longitude,
          nearestPoint.distanceKm,
          severity === "critical" ? "Route adjustment recommended and monitor ice conditions." : "Maintain the current route and monitor ice conditions.",
        ),
        metadata: {
          concentrationPercent: +(nearestPoint.concentration * 100).toFixed(1),
          forecastHorizonHours: input.seaIce.horizon_hours,
          model: input.seaIce.model,
        },
      });
    }
  } else {
    alerts.push(alertBase("sea-ice", "info", "Sea-Ice Alert Unavailable", "Sea-ice alert unavailable - data source unavailable.", input, undefined, undefined, undefined, "Retry when the sea-ice data source is available."));
  }

  if (input.icebergs.length) {
    const nearest = input.icebergs
      .map((iceberg) => ({ iceberg, distanceKm: distanceToRoute([iceberg.latitude, iceberg.longitude], coordinates) }))
      .sort((a, b) => a.distanceKm - b.distanceKm)[0];
    if (nearest) {
      const severity = icebergSeverity(nearest.distanceKm, input.hazardConfig);
      alerts.push({
        ...alertBase(
          "iceberg",
          severity,
          "Nearest Iceberg Alert",
          `${severity.toUpperCase()}: ${nearest.iceberg.iceberg_id} detected ${nearest.distanceKm.toFixed(1)} km from the ${input.routeLabel} route.`,
          input,
          nearest.iceberg.latitude,
          nearest.iceberg.longitude,
          nearest.distanceKm,
          severity === "critical" ? "Immediate clearance review recommended." : "Monitor iceberg movement and maintain safe clearance.",
        ),
        icebergId: nearest.iceberg.iceberg_id,
        metadata: {
          lengthKm: nearest.iceberg.length_km,
          widthKm: nearest.iceberg.width_km,
          lastObserved: nearest.iceberg.last_observed,
        },
      });
    }
  } else {
    alerts.push(alertBase("iceberg", "info", "Iceberg Alert Unavailable", "No nearby iceberg detected for the selected route.", input, undefined, undefined, undefined, "Continue monitoring the selected route."));
  }

  return alerts.sort((a, b) => {
    const severityOrder: Record<AlertSeverity, number> = { critical: 0, high: 1, warning: 2, medium: 2, low: 3, info: 4 };
    return severityOrder[a.severity] - severityOrder[b.severity] || (a.distanceKm ?? Infinity) - (b.distanceKm ?? Infinity);
  });
}
