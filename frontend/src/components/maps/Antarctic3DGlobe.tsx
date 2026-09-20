import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Globe, { type GlobeMethods } from "react-globe.gl";
import type { IcebergInfo, IcebergTrajectoryResponse } from "../../types";
import { concentrationColor } from "../../utils/colormap";
import { haversineKm } from "../../utils/haversine";
import { useLandMask, buildSeaIcePoints, isLandCell } from "./globeData";
import { GlobeControls } from "./GlobeControls";

export interface GridData {
  lat: number[];
  lon: number[];
  concentration: number[][];
}

interface SeaIcePoint {
  lat: number;
  lng: number;
  concentration: number;
}

interface IcebergPoint {
  id: string;
  lat: number;
  lng: number;
  radius: number;
  altitude: number;
  color: string;
  label: string;
  selected: boolean;
}

interface PathDatum {
  points: [number, number][];
  color: string;
  dash: number;
  stroke: number;
}

const MAX_SEAICE_POINTS = 14000;
const MAX_ICE_POINTS = 12000;

const SELECTED_LAT_MIN = -75;
const SELECTED_LAT_MAX = -55;

const ICE_COLOR = "#0ea5e9";
const ICE_SELECTED = "#ef4444";
const VESSEL_COLOR = "#10b981";

function useSize(el: HTMLDivElement | null): { width: number; height: number } {
  const [size, setSize] = useState({ width: 800, height: 400 });
  useLayoutEffect(() => {
    if (!el) return;
    const update = () =>
      setSize({ width: el.clientWidth || 800, height: el.clientHeight || 400 });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return size;
}

/** Small circle of points at a fixed latitude (used to subtly mark the 55–75°S band). */
function ringPoints(lat: number, n = 72): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const lng = (i / n) * 360 - 180;
    pts.push([lat, lng]);
  }
  return pts;
}

/** Mean drift speed (km/h) between consecutive timestamped observations, if derivable. */
function avgDriftKmph(trajectory: IcebergTrajectoryResponse | null | undefined): number | null {
  if (!trajectory) return null;
  const pts = trajectory.observations.map((o) => ({
    lat: o.latitude,
    lng: o.longitude,
    ts: o.timestamp ? Date.parse(o.timestamp) : NaN,
  }));
  let total = 0;
  let cnt = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    if (!Number.isFinite(a.ts) || !Number.isFinite(b.ts)) continue;
    const dtH = (b.ts - a.ts) / 3600000;
    if (dtH <= 0) continue;
    total += haversineKm(a.lat, a.lng, b.lat, b.lng) / dtH;
    cnt++;
  }
  return cnt ? total / cnt : null;
}

export interface Antarctic3DGlobeProps {
  seaIce?: GridData | null;
  seaIceTimestamp?: string | null;
  seaIceHorizon?: number | null;
  icebergs?: IcebergInfo[];
  selectedIceberg?: string | null;
  trajectory?: IcebergTrajectoryResponse | null;
  vesselPos?: [number, number] | null;
  vesselLabel?: string;
  onIcebergSelect?: (id: string) => void;
  height?: string;
}

export function Antarctic3DGlobe(props: Antarctic3DGlobeProps) {
  const {
    seaIce,
    seaIceTimestamp,
    seaIceHorizon,
    icebergs = [],
    selectedIceberg,
    trajectory,
    vesselPos,
    vesselLabel = "Vessel",
    onIcebergSelect,
    height = "100%",
  } = props;

  const globeRef = useRef<GlobeMethods | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const { width, height: pxHeight } = useSize(containerRef.current);

  const maskState = useLandMask();
  const mask = maskState.status === "ready" ? maskState.mask : null;
  const isLand = useCallback(
    (lat: number, lon: number) => (mask ? isLandCell(mask, lat, lon) : false),
    [mask],
  );

  const [webglOk, setWebglOk] = useState(true);
  useEffect(() => {
    try {
      const c = document.createElement("canvas");
      const gl =
        c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl");
      setWebglOk(Boolean(gl));
    } catch {
      setWebglOk(false);
    }
  }, []);

  const seaIcePoints = useMemo<SeaIcePoint[]>(() => {
    if (!seaIce || !seaIce.lat?.length || !seaIce.concentration?.length) return [];
    return buildSeaIcePoints(seaIce, isLand, MAX_SEAICE_POINTS);
  }, [seaIce, isLand]);

  const inBand = useCallback((lat: number) => {
    return lat >= SELECTED_LAT_MIN && lat <= SELECTED_LAT_MAX;
  }, []);

  const icebergPoints = useMemo<IcebergPoint[]>(() => {
    const drift = avgDriftKmph(trajectory);
    const pts = icebergs
      .filter(
        (ib) =>
          !ib.on_land &&
          !isLand(ib.latitude, ib.longitude) &&
          inBand(ib.latitude) &&
          Number.isFinite(ib.latitude) &&
          Number.isFinite(ib.longitude),
      )
      .map((ib): IcebergPoint => {
        const selected = selectedIceberg === ib.iceberg_id;
        const dims =
          ib.length_km != null
            ? `<div>${ib.length_km}${ib.width_km != null ? `×${ib.width_km}` : ""} km</div>`
            : "";
        const timestamp = ib.last_observed
          ? `<div>${new Date(ib.last_observed).toLocaleString()}</div>`
          : "";
        const speed = selected && drift != null ? `<div>Drift ≈ ${drift.toFixed(2)} km/h</div>` : "";
        const demo = ib.demo ? `<div style="color:#b45309">Demo</div>` : "";
        return {
          id: ib.iceberg_id,
          lat: ib.latitude,
          lng: ib.longitude,
          radius: selected ? 1.2 : 0.8,
          altitude: 0.03,
          color: selected ? ICE_SELECTED : ICE_COLOR,
          label: `<div style="font-size:12px;line-height:1.4">
              <strong>${ib.iceberg_id}</strong>
              <div>${ib.latitude.toFixed(3)}, ${ib.longitude.toFixed(3)}</div>
              ${timestamp}${dims}${speed}${demo}
            </div>`,
          selected,
        };
      });
    return pts.slice(0, MAX_ICE_POINTS);
  }, [icebergs, selectedIceberg, inBand, isLand, trajectory]);

  const vesselPoints = useMemo<IcebergPoint[]>(() => {
    if (!vesselPos) return [];
    return [
      {
        id: "vessel",
        lat: vesselPos[0],
        lng: vesselPos[1],
        radius: 1.1,
        altitude: 0.035,
        color: VESSEL_COLOR,
        label: `<div style="font-size:12px;line-height:1.4"><strong>${vesselLabel}</strong><div>${vesselPos[0].toFixed(3)}, ${vesselPos[1].toFixed(3)}</div></div>`,
        selected: false,
      },
    ];
  }, [vesselPos, vesselLabel]);

  const pointData = useMemo(
    () => [...seaIcePoints, ...icebergPoints, ...vesselPoints],
    [seaIcePoints, icebergPoints, vesselPoints],
  );

  const pathData = useMemo<PathDatum[]>(() => {
    const paths: PathDatum[] = [];
    // Subtle rings marking the selected 55–75°S Antarctic band on the sea-ice view
    if (seaIce) {
      paths.push({
        points: ringPoints(SELECTED_LAT_MAX, 96),
        color: "rgba(109,40,217,0.35)",
        dash: 4,
        stroke: 0.5,
      });
      paths.push({
        points: ringPoints(SELECTED_LAT_MIN, 96),
        color: "rgba(109,40,217,0.35)",
        dash: 4,
        stroke: 0.5,
      });
    }
    if (!trajectory) return paths;
    const obs: [number, number][] = trajectory.observations.map((p) => [
      p.latitude,
      p.longitude,
    ]);
    const pred: [number, number][] = trajectory.predictions.map((p) => [
      p.latitude,
      p.longitude,
    ]);
    if (obs.length > 1) {
      paths.push({ points: obs, color: "#6d28d9", dash: 0, stroke: 0.9 });
    }
    const connectedPrediction = obs.length > 0 ? [obs[obs.length - 1], ...pred] : pred;
    if (connectedPrediction.length > 1) {
      paths.push({ points: connectedPrediction, color: "#f59e0b", dash: 3, stroke: 1.0 });
    }
    return paths;
  }, [seaIce, trajectory]);

  const onPointClick = useCallback(
    (point: object) => {
      const p = point as SeaIcePoint | IcebergPoint;
      if ("id" in p && p.id !== "vessel" && onIcebergSelect) {
        onIcebergSelect(p.id);
      }
    },
    [onIcebergSelect],
  );

  // --- camera control ---
  // Full-Earth default view: whole globe in frame, classic map angle,
  // Antarctica visible near the bottom edge.
  const HOME_POV = useMemo(() => ({ lat: 25, lng: 15, altitude: 2.7 }), []);

  const flyToIceberg = useCallback(
    (id: string | null | undefined) => {
      if (!id || !globeRef.current) return;
      const ib = icebergs.find((i) => i.iceberg_id === id);
      if (!ib) return;
      globeRef.current.pointOfView(
        { lat: ib.latitude, lng: ib.longitude, altitude: 1.3 },
        900,
      );
    },
    [icebergs],
  );

  useEffect(() => {
    if (selectedIceberg) flyToIceberg(selectedIceberg);
  }, [selectedIceberg, flyToIceberg]);

  const zoom = useCallback((factor: number) => {
    const g = globeRef.current;
    if (!g) return;
    const pov = g.pointOfView();
    g.pointOfView(
      { lat: pov.lat, lng: pov.lng, altitude: Math.max(0.5, pov.altitude * factor) },
      350,
    );
  }, []);

  const resetView = useCallback(() => {
    globeRef.current?.pointOfView(HOME_POV, 700);
  }, [HOME_POV]);

  const toggleRotate = useCallback((on: boolean) => {
    const g = globeRef.current;
    if (!g) return;
    const c = g.controls();
    c.autoRotate = on;
    c.autoRotateSpeed = 0.6;
  }, []);

  const pointLatFn = (d: object) => (d as SeaIcePoint | IcebergPoint).lat;
  const pointLngFn = (d: object) => (d as SeaIcePoint | IcebergPoint).lng;
  const pointAltFn = (d: object) => {
    const p = d as SeaIcePoint | IcebergPoint;
    return "altitude" in p ? p.altitude : 0.025;
  };
  const pointColorFn = (d: object) => {
    const p = d as SeaIcePoint | IcebergPoint;
    if ("concentration" in p) return concentrationColor(p.concentration);
    return p.color;
  };
  const pointRadiusFn = (d: object) => {
    const p = d as SeaIcePoint | IcebergPoint;
    if ("concentration" in p) return 0.3;
    return p.radius;
  };
  const pointLabelFn = (d: object) => {
    const p = d as SeaIcePoint | IcebergPoint;
    if ("concentration" in p) {
      const ts = seaIceTimestamp ? new Date(seaIceTimestamp).toLocaleString() : "—";
      const horizon = seaIceHorizon != null ? `${seaIceHorizon}h` : "—";
      return `<div style="font-size:11px;line-height:1.4">
          <div>Lat ${p.lat.toFixed(2)}° · Lon ${p.lng.toFixed(2)}°</div>
          <div style="font-weight:600">${(p.concentration * 100).toFixed(1)}% concentration</div>
          <div>${ts}</div>
          <div>Horizon ${horizon}</div>
        </div>`;
    }
    return p.label;
  };
  const pathPointsFn = (d: object) => (d as PathDatum).points;
  const pathColorFn = (d: object) => (d as PathDatum).color;
  const pathStrokeFn = (d: object) => (d as PathDatum).stroke;
  const pathDashLengthFn = (d: object) => (d as PathDatum).dash;
  const pathDashGapFn = (d: object) => ((d as PathDatum).dash > 0 ? 1.5 : 0);

  if (!webglOk) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border border-slate-200 bg-white text-xs text-navy-500"
        style={{ height }}>
        WebGL is required to render the 3D globe but is unavailable in this browser.
      </div>
    );
  }

  return (
    <div className="relative w-full overflow-hidden rounded-xl border border-slate-200" style={{ height }}>
      <div ref={containerRef} className="h-full w-full">
        {pointData.length > 0 ? (
          <Globe
            ref={globeRef}
            width={width}
            height={pxHeight}
            backgroundColor="rgba(255,255,255,0)"
            globeImageUrl={`${import.meta.env.BASE_URL ?? ""}data/world-map.png`}
            atmosphereColor="#a7d7f5"
            atmosphereAltitude={0.12}
            onGlobeReady={() => globeRef.current?.pointOfView(HOME_POV, 0)}
            pointsData={pointData}
            pointLat={pointLatFn}
            pointLng={pointLngFn}
            pointAltitude={pointAltFn}
            pointColor={pointColorFn}
            pointRadius={pointRadiusFn}
            pointLabel={pointLabelFn}
            onPointClick={onPointClick}
            pathsData={pathData}
            pathPoints={pathPointsFn}
            pathColor={pathColorFn}
            pathStroke={pathStrokeFn}
            pathDashLength={pathDashLengthFn}
            pathDashGap={pathDashGapFn}
            pathPointAlt={0.04}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-xs text-navy-400">
            No data to display
          </div>
        )}
      </div>

      <GlobeControls
        onZoomIn={() => zoom(0.6)}
        onZoomOut={() => zoom(1.7)}
        onReset={resetView}
        onToggleRotate={toggleRotate}
      />

      {mask && (
        <div className="absolute bottom-2 right-2 z-[600] rounded bg-white/85 px-2 py-1 text-[10px] text-navy-500 shadow-card border border-slate-200">
          Ocean-only · land-mask verified
        </div>
      )}
      {maskState.status === "error" && (
        <div className="absolute bottom-2 left-2 z-[600] rounded bg-white/95 px-2 py-1 text-[10px] text-amber-700 shadow-card border border-amber-200 max-w-[260px]">
          Land mask unavailable — ocean-only rendering cannot be verified.
        </div>
      )}
      {!mask && maskState.status !== "error" && (
        <div className="absolute bottom-2 left-2 z-[600] rounded bg-white/85 px-2 py-1 text-[10px] text-navy-400 shadow-card border border-slate-200">
          Loading land mask…
        </div>
      )}
    </div>
  );
}