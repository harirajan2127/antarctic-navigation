import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import type {
  IcebergsListResponse,
  JourneyResponse,
  JourneyStatus,
  LegacyVesselInfo,
  PortInfo,
  ResearchCenterInfo,
  RouteResult,
  RoutesOptimizeResponse,
  SeaIceForecastResponse,
  SimulationConfig,
  VesselInfo,
} from "../types";
import { Card, CardHeader, CardBody } from "../components/common/Card";
import { useApp } from "../context/AppContext";
import { AntarcticMap } from "../components/map/AntarcticMap";
import { MetricBar, type JourneyMetric } from "../components/journey/MetricBar";
import { JourneyTimeline, type TimelineStep } from "../components/journey/JourneyTimeline";
import { calculateRouteAlerts } from "../utils/routeAlerts";

type Mode = "outbound" | "return";

function fmtLat(lat: number): string {
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}`;
}
function fmtLon(lon: number): string {
  return `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
}
function coords(lat: number, lon: number): string {
  return `${fmtLat(lat)}, ${fmtLon(lon)}`;
}

function parseCoordinate(value: string, axis: "lat" | "lon"): number | null {
  const match = value.trim().match(/^([+-]?\d+(?:\.\d+)?)\s*(NORTH|SOUTH|EAST|WEST|N|S|E|W)?$/i);
  if (!match) return null;
  const magnitude = Number(match[1]);
  const suffix = match[2]?.toUpperCase();
  const hemisphere = suffix?.[0];
  const isLatitude = axis === "lat";
  if (suffix && ((isLatitude && !["N", "S"].includes(hemisphere)) || (!isLatitude && !["E", "W"].includes(hemisphere)))) {
    return null;
  }
  const valueWithSign = hemisphere === "S" || hemisphere === "W" ? -Math.abs(magnitude) : magnitude;
  const limit = isLatitude ? 90 : 180;
  return valueWithSign >= -limit && valueWithSign <= limit ? valueWithSign : null;
}

function riskTone(level: string | null | undefined): JourneyMetric["tone"] {
  if (!level) return "normal";
  if (level === "low") return "good";
  if (level === "moderate") return "warn";
  return "bad";
}

function journeyToRoute(j: JourneyResponse): RouteResult {
  return {
    waypoints: j.waypoints.map((w) => ({
      lat: w.latitude,
      lon: w.longitude,
      risk_score: 0,
      step: 0,
    })),
    coordinates: j.waypoints.map((w) => [w.latitude, w.longitude] as [number, number]),
    distance_km: j.total_distance_nm * 1.852,
    distance_nm: j.total_distance_nm,
    travel_time_hours: j.estimated_duration_hours,
    fuel_tons: j.total_fuel_tons,
    risk_score: null,
    risk_level: j.max_risk_level,
    warnings: [],
  };
}

export function NavigationPlannerPage() {
  const { toast, setActiveRoute, replaceRouteAlerts, navigation, setNavigation } = useApp();

  const [ports, setPorts] = useState<PortInfo[]>([]);
  const [centers, setCenters] = useState<ResearchCenterInfo[]>([]);
  const [vessels, setVessels] = useState<VesselInfo[]>([]);
  const [legacyVessels, setLegacyVessels] = useState<LegacyVesselInfo[]>([]);
  const [simulationConfig, setSimulationConfig] = useState<SimulationConfig | null>(null);
  const [seaIce, setSeaIce] = useState<SeaIceForecastResponse | null>(null);
  const [icebergs, setIcebergs] = useState<IcebergsListResponse | null>(null);

  const { portId, centerId, vesselId, mode, journey, status, planning, selectedRoute, alertFocus, liveLat, liveLon } = navigation;
  const [mapRefreshToken, setMapRefreshToken] = useState(0);

  const [loadingMeta, setLoadingMeta] = useState(true);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [busy, setBusy] = useState<null | "start" | "advance" | "recalc">(null);
  const setPortId = (value: string) => setNavigation({ portId: value });
  const setCenterId = (value: string) => setNavigation({ centerId: value });
  const setVesselId = (value: string) => setNavigation({ vesselId: value });
  const setMode = (value: Mode) => setNavigation({ mode: value });
  const setJourney = (value: JourneyResponse | null) => setNavigation({ journey: value });
  const setStatus = (value: JourneyStatus | null) => setNavigation({ status: value });
  const setPlanning = (value: RoutesOptimizeResponse | null) => setNavigation({ planning: value });
  const setLiveLat = (value: string) => setNavigation({ liveLat: value });
  const setLiveLon = (value: string) => setNavigation({ liveLon: value });

  useEffect(() => {
    Promise.all([
      api.ports(),
      api.researchCenters(),
      api.legacyVessels(),
      api.vessels(),
      api.simulationConfig(),
      api.seaIceForecast(24),
      api.icebergs(),
    ])
      .then(([p, c, lv, v, sim, si, ib]) => {
        setPorts(p);
        setCenters(c);
        setLegacyVessels(lv);
        setVessels(v.vessels);
        setSimulationConfig(sim);
        setSeaIce(si);
        setIcebergs(ib);
        if (p.length && !navigation.portId) setPortId(p[0].port_id);
        if (c.length && !navigation.centerId) setCenterId(c[0].center_id);
        if (v.vessels.length && !navigation.vesselId) setVesselId(v.vessels[0].vessel_id);
      })
      .catch(() => toast("Failed to load navigation data", "error"))
      .finally(() => setLoadingMeta(false));
  }, [toast]);

  const port = useMemo(() => ports.find((x) => x.port_id === portId) ?? null, [ports, portId]);
  const center = useMemo(
    () => centers.find((x) => x.center_id === centerId) ?? null,
    [centers, centerId],
  );
  const vessel = useMemo(
    () => vessels.find((x) => x.vessel_id === vesselId) ?? null,
    [vessels, vesselId],
  );

  const fetchStatus = useCallback(async (id: string) => {
    try {
      setStatus(await api.journeyStatus(id));
    } catch {
      setStatus(null);
    }
  }, []);

  const fetchAlternatives = useCallback(
    async (fromPort: PortInfo, toCenter: ResearchCenterInfo, vId: string) => {
      try {
        const plan = await api.routesOptimize({
          start_latitude: fromPort.latitude,
          start_longitude: fromPort.longitude,
          destination_latitude: toCenter.latitude,
          destination_longitude: toCenter.longitude,
          vessel_id: vId,
          preference: "recommended",
        });
        setRouteError(null);
        setPlanning(plan);
        setNavigation({ selectedRoute: plan.recommended });
        setActiveRoute(plan);
      } catch (e) {
        setPlanning(null); // alternatives unavailable — never shown as fake
        setRouteError(e instanceof Error ? e.message : "Route generation failed.");
      }
    },
    [setActiveRoute, setNavigation],
  );

  useEffect(() => {
    if (!selectedRoute || !planning) {
      replaceRouteAlerts([]);
      return;
    }
    replaceRouteAlerts(calculateRouteAlerts({
      route: selectedRoute,
      routeId: planning.route_id,
      routeLabel: selectedRoute === planning.recommended ? "Recommended Route" : `${selectedRoute.risk_level ?? "Alternative"} Risk Route`,
      seaIce,
      icebergs: icebergs?.icebergs ?? [],
      hazardConfig: planning.hazard_config,
    }));
  }, [selectedRoute, planning, seaIce, icebergs, mapRefreshToken, replaceRouteAlerts]);

  const startJourney = useCallback(async () => {
    if (!port || !center) {
      toast("Select a departure port and a research center", "error");
      return;
    }
    setBusy("start");
    try {
      const j = await api.startJourney({
        departure_port_id: port.port_id,
        destination_id: center.center_id,
        vessel_id: vesselId,
        journey_mode: mode,
        position_mode: "simulation",
      });
      setJourney(j);
      setNavigation({ selectedRoute: journeyToRoute(j) });
      await fetchStatus(j.journey_id);
      await fetchAlternatives(port, center, vesselId);
      toast(`Journey started — ${j.origin} → ${j.destination} (${j.journey_mode})`, "success");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Failed to start journey", "error");
    } finally {
      setBusy(null);
    }
  }, [port, center, vesselId, mode, fetchStatus, fetchAlternatives, toast]);

  const advance = useCallback(async () => {
    if (!journey) {
      toast("Start a journey first", "error");
      return;
    }
    setBusy("advance");
    try {
      const j = await api.advanceJourney(journey.journey_id);
      setJourney(j);
      setNavigation({ selectedRoute: journeyToRoute(j) });
      await fetchStatus(j.journey_id);
      toast(`Advanced 2 simulated hours — route update #${j.route_update_count}`, "success");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Failed to advance", "error");
    } finally {
      setBusy(null);
    }
  }, [journey, fetchStatus, toast]);

  const recalc = useCallback(async () => {
    if (!journey) {
      toast("Start a journey first", "error");
      return;
    }
    const lat = parseCoordinate(liveLat, "lat");
    const lon = parseCoordinate(liveLon, "lon");
    if (lat === null || lon === null) {
      toast("Use coordinates such as 31S and 79E", "error");
      return;
    }
    setBusy("recalc");
    try {
      const j = await api.recalculateJourney(journey.journey_id, lat, lon);
      setJourney(j);
      setNavigation({ selectedRoute: journeyToRoute(j) });
      await fetchStatus(j.journey_id);
      toast(`Route recalculated from ${coords(lat, lon)}`, "success");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Failed to recalculate", "error");
    } finally {
      setBusy(null);
    }
  }, [journey, liveLat, liveLon, fetchStatus, toast]);

  const resetDetails = useCallback(() => {
    setNavigation({ journey: null, status: null, planning: null, selectedRoute: null, liveLat: "", liveLon: "" });
    setActiveRoute(null);
    setLiveLat("");
    setLiveLon("");
    toast("Journey details reset", "info");
  }, [toast]);

  const refreshMap = useCallback(() => {
    setMapRefreshToken((token) => token + 1);
  }, []);

  const stepHours = simulationConfig?.time_step_hours ?? 2;

  const journeyRoute = useMemo(
    () => selectedRoute ?? (journey ? journeyToRoute(journey) : null),
    [journey, selectedRoute],
  );

  const vesselPos: [number, number] | null = useMemo(() => {
    if (journey) return [journey.current_vessel_lat, journey.current_vessel_lon];
    if (planning) return [planning.start_latitude, planning.start_longitude];
    if (port) return [port.latitude, port.longitude];
    return null;
  }, [journey, planning, port]);

  const nowHours = status?.current_time_hours ?? 0;
  const lastUpdateCount = journey?.route_update_count ?? null;

  const metrics: JourneyMetric[] = [
    {
      label: "Current Vessel Location",
      value: journey ? coords(journey.current_vessel_lat, journey.current_vessel_lon) : vesselPos ? coords(vesselPos[0], vesselPos[1]) : "—",
      sub: vessel?.name ?? null,
    },
    {
      label: "Destination",
      value: journey?.destination ?? center?.name ?? "—",
      sub: center ? coords(center.latitude, center.longitude) : null,
    },
    {
      label: "Journey Mode",
      value: journey ? journey.journey_mode : mode,
      sub: "simulated position",
    },
    {
      label: "Last Update",
      value: journey ? `T+${nowHours.toFixed(1)} h` : "—",
      sub: lastUpdateCount != null ? `update #${lastUpdateCount}` : null,
    },
    {
      label: "Next Update",
      value: journey ? `T+${(nowHours + stepHours).toFixed(1)} h` : "—",
      sub: `every ${stepHours} h`,
    },
    {
      label: "Remaining Distance",
      value: status ? `${status.remaining_distance_nm.toFixed(1)} nm` : journey ? `${journey.total_distance_nm.toFixed(1)} nm` : planning ? `${planning.recommended.distance_nm.toFixed(1)} nm` : "—",
      sub: status && !status.is_complete ? `${status.progress_percent.toFixed(0)}% complete` : null,
    },
    {
      label: "Fuel Estimate",
      value: journey ? `${journey.total_fuel_tons.toFixed(1)} t` : planning?.recommended?.fuel_tons != null ? `${planning.recommended.fuel_tons.toFixed(1)} t` : "—",
      sub: null,
    },
    {
      label: "Risk Level",
      value: journey?.max_risk_level ?? planning?.recommended.risk_level ?? "—",
      tone: riskTone(journey?.max_risk_level ?? planning?.recommended.risk_level),
      sub: null,
    },
  ];

  const timeline: TimelineStep[] = [
    {
      key: "departure",
      label: "Departure",
      value: port?.name ?? "—",
      detail: port ? coords(port.latitude, port.longitude) : null,
      state: "done",
    },
    {
      key: "prev-update",
      label: "Previous 2h Update",
      value: lastUpdateCount && lastUpdateCount > 1 ? `T+${Math.max(0, nowHours - stepHours).toFixed(1)} h` : "None yet",
      detail: lastUpdateCount && lastUpdateCount > 1 ? `update #${lastUpdateCount - 1}` : "awaiting first advance",
      state: lastUpdateCount && lastUpdateCount > 1 ? "done" : "upcoming",
    },
    {
      key: "current-pos",
      label: "Current Position",
      value: journey ? `T+${nowHours.toFixed(1)} h` : "At berth",
      detail: vesselPos ? coords(vesselPos[0], vesselPos[1]) : null,
      state: "active",
    },
    {
      key: "current-route",
      label: "Current Route",
      value: journey ? `${journey.total_distance_nm.toFixed(0)} nm` : planning ? `${planning.recommended.distance_nm.toFixed(0)} nm` : "Not planned",
      detail: journey ? `update #${journey.route_update_count} · ${journey.max_risk_level} risk` : null,
      state: journey ? "active" : "upcoming",
    },
    {
      key: "next-update",
      label: "Next Update",
      value: journey ? `T+${(nowHours + stepHours).toFixed(1)} h` : "—",
      detail: journey ? `route recompute (${stepHours} h cycle)` : null,
      state: "upcoming",
    },
    {
      key: "destination",
      label: "Destination",
      value: journey?.destination ?? center?.name ?? "—",
      detail: center ? coords(center.latitude, center.longitude) : null,
      state: status?.is_complete ? "done" : "upcoming",
    },
  ];

  const selectCls =
    "w-full text-xs border border-slate-200 rounded-lg px-2 py-2 bg-white text-navy-700 focus:outline-none focus:ring-1 focus:ring-blue-500";

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-bold text-navy-900">Navigation Dashboard</h1>
          <p className="text-xs text-navy-400 mt-0.5">
            Journey monitoring · route planning · 2-hour rolling updates
          </p>
        </div>
        {seaIce?.warning && (
          <span className="px-2 py-1 text-[11px] font-medium bg-amber-50 border border-amber-200 text-amber-700 rounded-lg">
            {seaIce.warning}
          </span>
        )}
      </div>

      <MetricBar metrics={metrics} />

      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card>
          <CardHeader
            title="Journey Controls"
            subtitle="Start a simulation, then advance every 2 h"
            action={
              <div className="flex items-center gap-2">
                {legacyVessels.length > 0 && (
                  <span className="text-[10px] text-navy-400">{legacyVessels.length} vessels</span>
                )}
                <button
                  type="button"
                  onClick={resetDetails}
                  disabled={busy !== null}
                  className="px-2 py-1 text-[10px] font-semibold border border-slate-300 text-navy-600 rounded-md hover:bg-slate-50 disabled:opacity-50"
                >
                  Reset Details
                </button>
              </div>
            }
          />
          <CardBody className="space-y-3">
            <div>
              <label className="block text-[10px] font-semibold uppercase tracking-wider text-navy-500 mb-1">
                Departure Port
              </label>
              <select
                value={portId}
                onChange={(e) => setPortId(e.target.value)}
                disabled={loadingMeta || !!journey}
                className={selectCls}
              >
                {ports.map((p) => (
                  <option key={p.port_id} value={p.port_id}>
                    {p.name} · {p.country}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-semibold uppercase tracking-wider text-navy-500 mb-1">
                Research Center
              </label>
              <select
                value={centerId}
                onChange={(e) => setCenterId(e.target.value)}
                disabled={loadingMeta || !!journey}
                className={selectCls}
              >
                {centers.map((c) => (
                  <option key={c.center_id} value={c.center_id}>
                    {c.name} · {c.country}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-semibold uppercase tracking-wider text-navy-500 mb-1">
                  Vessel
                </label>
                <select
                  value={vesselId}
                  onChange={(e) => setVesselId(e.target.value)}
                  disabled={loadingMeta || !!journey}
                  className={selectCls}
                >
                  {vessels.map((v) => (
                    <option key={v.vessel_id} value={v.vessel_id}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[10px] font-semibold uppercase tracking-wider text-navy-500 mb-1">
                  Journey Mode
                </label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value as Mode)}
                  disabled={!!journey}
                  className={selectCls}
                >
                  <option value="outbound">Outbound</option>
                  <option value="return">Return</option>
                </select>
              </div>
            </div>

            <button
              onClick={startJourney}
              disabled={busy !== null || loadingMeta}
              className="w-full px-3 py-2.5 text-sm font-semibold bg-navy-800 text-white rounded-lg hover:bg-navy-900 transition-colors disabled:opacity-50"
            >
              {busy === "start" ? "Starting…" : journey ? "Journey Active" : "Start Journey"}
            </button>

            <button
              onClick={advance}
              disabled={busy !== null || !journey || status?.is_complete}
              className="w-full px-3 py-2.5 text-sm font-semibold bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {busy === "advance" ? "Advancing…" : "Advance 2 Hours"}
            </button>

            <div className="pt-2 border-t border-slate-100">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-navy-400 mb-2">
                Recalculate from fix (live position)
              </p>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="text"
                  step="any"
                  value={liveLat}
                  onChange={(e) => setLiveLat(e.target.value)}
                  disabled={!journey}
                  placeholder="Latitude (31S)"
                  className="w-full text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-navy-700"
                />
                <input
                  type="text"
                  step="any"
                  value={liveLon}
                  onChange={(e) => setLiveLon(e.target.value)}
                  disabled={!journey}
                  placeholder="Longitude (79E)"
                  className="w-full text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-navy-700"
                />
              </div>
              <button
                onClick={recalc}
                disabled={busy !== null || !journey}
                className="mt-2 w-full px-3 py-2 text-xs font-semibold border border-blue-600 text-blue-700 rounded-lg hover:bg-blue-50 transition-colors disabled:opacity-50"
              >
                {busy === "recalc" ? "Recalculating…" : "Recalculate Route"}
              </button>
            </div>

            {status?.is_complete && (
              <p className="text-xs font-semibold text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2">
                Journey complete — arrived at {journey?.destination} after {nowHours.toFixed(1)} h.
              </p>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Voyage Map"
            subtitle={
              journey
                ? `${journey.origin} → ${journey.destination} · update #${journey.route_update_count}`
                : port
                  ? `${port.name} → ${center?.name ?? "select center"}`
                  : "Loading configuration…"
            }
            action={
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-3 text-[10px] text-navy-500">
                  <span className="flex items-center gap-1">
                    <span className="w-4 h-1.5 rounded bg-navy-800 inline-block" />
                    Recommended
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-4 border-t-2 border-dashed border-blue-400 inline-block" />
                    Alternative
                  </span>
                </div>
                <button
                  type="button"
                  onClick={refreshMap}
                  className="map-refresh-button"
                  aria-label="Refresh map"
                  title="Refresh map"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M20 11a8 8 0 0 0-14.7-4.3L3 9m0 0V4m0 5h5M4 13a8 8 0 0 0 14.7 4.3L21 15m0 0v5m0-5h-5" />
                  </svg>
                </button>
              </div>
            }
          />
          <CardBody>
            <div className="h-[420px] sm:h-[520px] xl:h-[560px] rounded-lg overflow-hidden border border-slate-200">
              <AntarcticMap
                seaIce={seaIce ?? null}
                icebergs={icebergs?.icebergs ?? []}
                recommended={journeyRoute}
                alternatives={planning?.alternatives ?? []}
                vesselPos={vesselPos}
                vesselLabel={vessel?.name ?? "Vessel"}
                startPoint={port ? { name: port.name, lat: port.latitude, lon: port.longitude } : null}
                endPoint={
                  center ? { name: center.name, lat: center.latitude, lon: center.longitude } : null
                }
                fitBounds={false}
                refreshToken={mapRefreshToken}
                focusPoint={alertFocus}
                height="100%"
              />
            </div>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Journey Timeline"
          subtitle="2-hour rolling route updates between departure and destination"
        />
        <CardBody>
          <div className="overflow-x-auto py-2">
            <JourneyTimeline steps={timeline} />
          </div>
        </CardBody>
      </Card>

      {routeError && (
        <Card>
          <CardBody>
            <div className="flex items-center justify-between gap-3 text-xs text-red-700">
              <span>Route generation failed. Reason: {routeError}</span>
              {port && center && (
                <button
                  type="button"
                  onClick={() => fetchAlternatives(port, center, vesselId)}
                  className="px-3 py-1.5 rounded-lg bg-red-50 hover:bg-red-100 font-medium"
                >
                  Retry
                </button>
              )}
            </div>
          </CardBody>
        </Card>
      )}

      {(journey || planning) ? (
        <Card>
          <CardHeader
            title="Route Details"
            subtitle={journey ? `journey ${journey.journey_id.slice(0, 8)}` : planning ? `route ${planning.route_id.slice(0, 8)} · planning only` : ""}
          />
          <CardBody>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {[
                {
                  label: "Distance",
                  value: journey
                    ? `${journey.total_distance_nm.toFixed(0)} nm`
                    : planning
                      ? `${planning.recommended.distance_nm.toFixed(0)} nm`
                      : "—",
                },
                {
                  label: "Est. Duration",
                  value: journey
                    ? `${journey.estimated_duration_hours.toFixed(1)} h`
                    : planning
                      ? `${planning.recommended.travel_time_hours.toFixed(1)} h`
                      : "—",
                },
                {
                  label: "Fuel",
                  value: journey
                    ? `${journey.total_fuel_tons.toFixed(1)} t`
                    : planning
                      ? `${(planning.recommended.fuel_tons ?? 0).toFixed(1)} t`
                      : "—",
                },
                {
                  label: "Risk",
                  value: journey?.max_risk_level ?? planning?.recommended.risk_level ?? "—",
                },
                {
                  label: "Waypoints",
                  value: `${journey?.waypoints.length ?? planning?.recommended.waypoints.length ?? 0}`,
                },
                {
                  label: "Route Updates",
                  value: `${journey?.route_update_count ?? 1}`,
                },
              ].map((s) => (
                <div key={s.label} className="p-3 bg-slate-50 rounded-lg text-center">
                  <p className="text-[10px] uppercase tracking-wider text-navy-400">{s.label}</p>
                  <p className="mt-1 text-sm font-bold text-navy-900">{s.value}</p>
                </div>
              ))}
            </div>

            {planning?.ocean_approach_distance_km != null && (
              <div className="mt-4 p-2 bg-amber-50 border border-amber-100 rounded-lg text-[11px] text-amber-700">
                {planning.destination_name ?? "Destination"} is on land. Routing to the
                nearest navigable ocean cell {planning.ocean_approach_distance_km.toFixed(0)} km
                away ({planning.destination_latitude.toFixed(2)},{" "}
                {planning.destination_longitude.toFixed(2)}). Station coordinates are
                preserved; this is the ocean approach point used for route generation.
              </div>
            )}

            {planning && planning.alternatives.length > 0 && (
              <div className="mt-4">
                <h4 className="text-xs font-semibold text-navy-700 mb-2">
                  Select route for map and alerts
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <button type="button" onClick={() => setNavigation({ selectedRoute: planning.recommended })} className={`w-full text-left p-2 border rounded-lg text-[11px] ${selectedRoute === planning.recommended ? "border-blue-400 bg-blue-50" : "border-slate-200 hover:bg-slate-50"}`}>
                    <p className="font-semibold text-navy-800">Recommended route</p>
                    <p className="text-navy-500">{planning.recommended.distance_nm.toFixed(0)} nm · {planning.recommended.risk_level ?? "—"}</p>
                  </button>
                  {planning.alternatives.map((alt, i) => (
                    <button key={i} type="button" onClick={() => setNavigation({ selectedRoute: alt })} className={`w-full text-left p-2 border rounded-lg text-[11px] ${selectedRoute === alt ? "border-blue-400 bg-blue-50" : "border-slate-200 hover:bg-slate-50"}`}>
                      <p className="font-semibold text-navy-800">
                        {alt.travel_time_hours <= (planning.recommended.travel_time_hours ?? 0) + 1 ? "Faster alternative" : "Alternative"} {i + 1}
                      </p>
                      <p className="text-navy-500">
                        {alt.distance_nm.toFixed(0)} nm · {alt.travel_time_hours.toFixed(1)} h ·{" "}
                        {alt.fuel_tons != null ? `${alt.fuel_tons.toFixed(1)} t` : "—"} · {alt.risk_level ?? "—"}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {planning?.disclaimer && (
              <div className="mt-3 p-2 bg-blue-50 border border-blue-100 rounded-lg text-[11px] text-blue-700">
                {planning.disclaimer}
              </div>
            )}
          </CardBody>
        </Card>
      ) : null}
    </div>
  );
}