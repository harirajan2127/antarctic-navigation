import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Rectangle,
  CircleMarker,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import type { IcebergInfo, RouteResult, IcebergTrajectoryResponse } from "../../types";
import { concentrationToCanvas, seaIceLegendStops } from "../../utils/colormap";

export interface GridData {
  lat: number[];
  lon: number[];
  concentration: number[][];
}

export interface PointLabel {
  name: string;
  lat: number;
  lon: number;
}

interface AntarcticMapProps {
  seaIce?: GridData | null;
  icebergs?: IcebergInfo[];
  selectedIceberg?: string | null;
  trajectory?: IcebergTrajectoryResponse | null;
  recommended?: RouteResult | null;
  alternatives?: RouteResult[];
  vesselPos?: [number, number] | null;
  vesselLabel?: string;
  startPoint?: PointLabel | null;
  endPoint?: PointLabel | null;
  fitBounds?: boolean;
  refreshToken?: number;
  focusPoint?: [number, number] | null;
  showSeaIceLegend?: boolean;
  height?: string;
}

const NAVY = "#0b3d6e";

// Dashboard display rule: the whole selected view is clipped to a horizontal
// coastal ocean band around Antarctica — 55°S to 75°S — and only to ocean.
//
//   * Iceberg markers : drawn ONLY when at sea AND latitude inside the band.
//     "Land" icebergs hang over the continent; the loader annotates each with
//     ``on_land``/``distance_to_coast_km`` so this view can hide them.
//   * Sea-ice overlay : clipped to the same band (ocean-only; the grid has no
//     land cells).
//   * A purple dashed rectangle marks the band boundary with 55°S/75°S labels
//     on the left and right edges.
//
// Display-only — it does not change counts, analytics, predictions, the
// 40-item iceberg list, trajectories, routes, the vessel, or sea-ice model
// behaviour assessments.
const SELECTED_LAT_MIN = -75; // 75°S — bottom (south) edge of the band
const SELECTED_LAT_MAX = -55; // 55°S — top (north) edge of the band
const SELECTED_BAND_COLOR = "#a855f7"; // purple dashed boundary + labels

// Small purple "lat chip" used for the 55°S / 75°S labels that sit on the left
// and right edges of the selected band.  Rendered by the CSS-only purple
// dashed border so it visually matches the band rectangle (no images).
function bandLabelIcon(text: string): L.DivIcon {
  return L.divIcon({
    className: "",
    html: `<div class="band-lat-chip" style="border-color:${SELECTED_BAND_COLOR};color:${SELECTED_BAND_COLOR}">${text}</div>`,
    iconSize: [44, 14],
    iconAnchor: [22, 8],
  });
}

function vesselIcon(): L.DivIcon {
  return L.divIcon({
    className: "",
    html: `<div class="vessel-marker"><span class="vessel-core"></span></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}

function namedLabelIcon(name: string, color: string): L.DivIcon {
  return L.divIcon({
    className: "",
    html: `<div class="map-label"><span class="map-label-dot" style="background:${color};box-shadow:0 0 0 2px ${color}55"></span><span>${name}</span></div>`,
    iconSize: [0, 0],
    iconAnchor: [10, 5],
  });
}

function formatCoordinate(value: number, positive: string, negative: string): string {
  return `${Math.abs(value).toFixed(4)}° ${value >= 0 ? positive : negative}`;
}

function CoordinateInteraction({
  onMove,
  onLeave,
  onTap,
}: {
  onMove: (latlng: L.LatLng, point: L.Point) => void;
  onLeave: () => void;
  onTap: (latlng: L.LatLng, point: L.Point) => void;
}) {
  const map = useMap();

  useEffect(() => {
    const handleMove = (event: L.LeafletMouseEvent) => onMove(event.latlng, event.containerPoint);
    const handleLeave = () => onLeave();
    const handleTap = (event: L.LeafletMouseEvent) => onTap(event.latlng, event.containerPoint);
    map.on("mousemove", handleMove);
    map.on("mouseout", handleLeave);
    map.on("click", handleTap);
    return () => {
      map.off("mousemove", handleMove);
      map.off("mouseout", handleLeave);
      map.off("click", handleTap);
    };
  }, [map, onLeave, onMove, onTap]);

  return null;
}

function GraticuleLayer({ visible }: { visible: boolean }) {
  const map = useMap();
  const layerRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    const layer = L.layerGroup().addTo(map);
    layerRef.current = layer;

    const redraw = () => {
      layer.clearLayers();
      if (!visible) return;
      const bounds = map.getBounds();
      const south = Math.max(-90, Math.floor(bounds.getSouth() / 10) * 10);
      const north = Math.min(90, Math.ceil(bounds.getNorth() / 10) * 10);
      const west = Math.max(-180, Math.floor(bounds.getWest() / 10) * 10);
      const east = Math.min(180, Math.ceil(bounds.getEast() / 10) * 10);
      const pathOptions: L.PathOptions = {
        color: "#64748b",
        weight: 1,
        opacity: 0.22,
        interactive: false,
      };
      for (let latitude = south; latitude <= north; latitude += 10) {
        L.polyline([[latitude, west], [latitude, east]], pathOptions).addTo(layer);
      }
      for (let longitude = west; longitude <= east; longitude += 10) {
        L.polyline([[south, longitude], [north, longitude]], pathOptions).addTo(layer);
      }
    };

    redraw();
    map.on("moveend zoomend", redraw);
    return () => {
      map.off("moveend zoomend", redraw);
      layer.remove();
      layerRef.current = null;
    };
  }, [map, visible]);

  return null;
}

function RefreshMapLayer({ refreshToken = 0 }: { refreshToken?: number }) {
  const map = useMap();

  useEffect(() => {
    if (!refreshToken) return;
    sessionStorage.removeItem("antarctic-map-view");
    map.invalidateSize({ animate: false });
    map.setView([-68, 80], 3.2, { animate: true });
  }, [map, refreshToken]);

  return null;
}

function PersistMapViewLayer() {
  const map = useMap();

  useEffect(() => {
    const save = () => {
      sessionStorage.setItem(
        "antarctic-map-view",
        JSON.stringify({ center: [map.getCenter().lat, map.getCenter().lng], zoom: map.getZoom() }),
      );
    };
    map.on("moveend", save);
    return () => {
      map.off("moveend", save);
    };
  }, [map]);

  return null;
}

function SeaIceLegend() {
  const stops = seaIceLegendStops();
  return (
    <div className="absolute bottom-3 right-3 z-[500] bg-white/95 border border-slate-200 rounded-lg shadow-card px-3 py-2 w-44 pointer-events-auto">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-navy-500 mb-1">
        Sea Ice Concentration
      </p>
      <div
        className="h-2 rounded-full"
        style={{
          background: `linear-gradient(to right, ${stops
            .map((s) => `${s.color} ${s.pct}%`)
            .join(", ")})`,
        }}
      />
      <div className="flex justify-between text-[9px] text-navy-400 mt-0.5">
        <span>0%</span>
        <span>15</span>
        <span>30</span>
        <span>50</span>
        <span>70</span>
          <span>100%</span>
        </div>

        <div className="mt-2 border-t border-slate-200 pt-1.5 space-y-1.5">
          <div className="flex items-center gap-1.5 text-[9px] text-navy-500">
            <span className="w-2.5 h-2.5 rounded-full border border-sky-500 bg-sky-400" />
            Icebergs (at sea)
          </div>
          <div className="flex items-center gap-1.5 text-[9px] font-semibold text-navy-600">
            <span
              className="block h-0 w-5 border-t-2 border-dashed"
              style={{ borderColor: SELECTED_BAND_COLOR }}
            />
            Selected Area · 55°S–75°S, ocean only
          </div>
        </div>
      </div>
    );
  }

function SeaIceOverlay({ seaIce }: { seaIce: GridData }) {
  const map = useMap();
  const overlayRef = useRef<L.ImageOverlay | null>(null);

  useEffect(() => {
    // Clip the raster to the selected band (55°S–75°S, ocean-only): keep only
    // the grid rows whose latitude falls inside the band substitute.  The grid
    // has no land cells, so slicing by latitude alone never draws over land.
    const latRowIdx: number[] = [];
    for (let i = 0; i < seaIce.lat.length; i++) {
      if (seaIce.lat[i] >= SELECTED_LAT_MIN && seaIce.lat[i] <= SELECTED_LAT_MAX) {
        latRowIdx.push(i);
      }
    }
    const clippedConcentration = latRowIdx.map((i) => seaIce.concentration[i]);

    const canvas = concentrationToCanvas(clippedConcentration);
    const dataUrl = canvas.toDataURL();
    const latMin = seaIce.lat[latRowIdx[0] ?? 0];
    const latMax = seaIce.lat[latRowIdx[latRowIdx.length - 1] ?? seaIce.lat.length - 1];
    const lonMin = seaIce.lon[0];
    const lonMax = seaIce.lon[seaIce.lon.length - 1];
    const bounds: L.LatLngBoundsExpression = [
      [latMin, lonMin],
      [latMax, lonMax],
    ];

    if (overlayRef.current) {
      overlayRef.current.setUrl(dataUrl);
      overlayRef.current.setBounds(L.latLngBounds(bounds));
    } else {
      overlayRef.current = L.imageOverlay(dataUrl, L.latLngBounds(bounds), {
        opacity: 0.6,
        interactive: false,
      }).addTo(map);
    }

    return () => {
      overlayRef.current?.remove();
      overlayRef.current = null;
    };
  }, [seaIce, map]);

  return null;
}

function TrajectoryLayer({ trajectory }: { trajectory: IcebergTrajectoryResponse }) {
  const obs = trajectory.observations.map(
    (p) => [p.latitude, p.longitude] as [number, number],
  );
  const pred = trajectory.predictions.map(
    (p) => [p.latitude, p.longitude] as [number, number],
  );
  const all = [...obs, ...pred];
  const connectedPrediction = obs.length > 0 ? [obs[obs.length - 1], ...pred] : pred;
  return (
    <>
      {obs.length > 1 && (
        <Polyline
          positions={obs}
          pathOptions={{ color: "#6d28d9", weight: 3, opacity: 0.9 }}
        />
      )}
      {connectedPrediction.length > 1 && (
        <Polyline
          positions={connectedPrediction}
          pathOptions={{
            color: "#f59e0b",
            weight: 3,
            opacity: 0.95,
            dashArray: "6 4",
          }}
        />
      )}
      {all.map((p, i) => (
        <CircleMarker
          key={i}
          center={p}
          radius={i < obs.length ? 3 : 4}
          pathOptions={{
            color: i < obs.length ? "#6d28d9" : "#f59e0b",
            fillColor: i < obs.length ? "#6d28d9" : "#f59e0b",
            fillOpacity: 1,
            weight: 0,
          }}
        />
      ))}
    </>
  );
}

function FitBoundsLayer({
  points,
  enabled,
}: {
  points: [number, number][];
  enabled: boolean;
}) {
  const map = useMap();
  const signature = useMemo(
    () =>
      points
        .map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`)
        .join("|"),
    [points],
  );

  useEffect(() => {
    if (!enabled || points.length === 0) return;
    // Fit bounds safely across the antimeridian. Raw [[lat, lon]] pairs from
    // the API can straddle 180° (e.g. one waypoint at +179.7 and another at
    // -179.7) — `L.latLngBounds` would then inflate to nearly the whole world.
    // Unwrap longitudes around the mid-point so the box spans the shortest
    // possible arc instead.
    const lats = points.map((p) => p[0]);
    const lons = points.map((p) => ((p[1] + 180) % 360) - 180);
    const lonMin = Math.min(...lons);
    const lonMax = Math.max(...lons);
    if (lonMax - lonMin > 270) {
      // Points are split across the antimeridian: shift everything < 0 up by
      // 360 so the step near 180° is continuous, then clamp back to [-180,180].
      const shifted = points.map(([la, lo]) => [
        la,
        lo < 0 ? lo + 360 : lo,
      ]);
      const slon = shifted.map((p) => p[1]);
      const slonMin = Math.min(...slon);
      const slonMax = Math.max(...slon);
      map.fitBounds(
        L.latLngBounds([
          { lat: Math.min(...lats), lng: Math.max(slonMin + 360, -180) > 180 ? 180 : slonMin + 360 },
          { lat: Math.min(...lats), lng: Math.max(slonMax, -180) },
          { lat: Math.max(...lats), lng: slonMin + 360 },
          { lat: Math.max(...lats), lng: slonMax },
        ]),
        { padding: [44, 44], maxZoom: 7 },
      );
    } else {
      map.fitBounds(L.latLngBounds(points), { padding: [44, 44], maxZoom: 7 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, enabled, map]);

  return null;
}

function FocusPointLayer({ point }: { point?: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (!point) return;
    map.flyTo(point, Math.max(map.getZoom(), 5), { duration: 0.6 });
  }, [map, point]);
  return null;
}

export function AntarcticMap(props: AntarcticMapProps) {
  const {
    seaIce,
    icebergs = [],
    selectedIceberg,
    trajectory,
    recommended,
    alternatives = [],
    vesselPos,
    vesselLabel = "Vessel",
    startPoint,
    endPoint,
    fitBounds = true,
    refreshToken = 0,
    focusPoint = null,
    showSeaIceLegend = true,
    height = "100%",
  } = props;
  const [cursorCoordinate, setCursorCoordinate] = useState<L.LatLng | null>(null);
  const [cursorPoint, setCursorPoint] = useState<L.Point | null>(null);
  const [touchPinned, setTouchPinned] = useState(false);
  const [graticuleVisible, setGraticuleVisible] = useState(false);
  const [copied, setCopied] = useState(false);
  const savedView = useMemo(() => {
    try {
      const raw = sessionStorage.getItem("antarctic-map-view");
      if (!raw) return null;
      const parsed = JSON.parse(raw) as { center?: [number, number]; zoom?: number };
      return parsed.center && typeof parsed.zoom === "number" ? parsed : null;
    } catch {
      return null;
    }
  }, []);

  const handleMapMove = useCallback((latlng: L.LatLng, point: L.Point) => {
    setCursorCoordinate(latlng);
    setCursorPoint(point);
    setTouchPinned(false);
    setCopied(false);
  }, []);
  const handleMapLeave = useCallback(() => {
    if (!touchPinned) {
      setCursorCoordinate(null);
      setCursorPoint(null);
    }
  }, [touchPinned]);
  const handleMapTap = useCallback((latlng: L.LatLng, point: L.Point) => {
    if (typeof navigator !== "undefined" && navigator.maxTouchPoints > 0) {
      setCursorCoordinate(latlng);
      setCursorPoint(point);
      setTouchPinned(true);
    }
  }, []);

  const coordinateText = cursorCoordinate
    ? `Latitude: ${formatCoordinate(cursorCoordinate.lat, "N", "S")}\nLongitude: ${formatCoordinate(cursorCoordinate.lng, "E", "W")}`
    : "";

  const copyCoordinates = async () => {
    if (!coordinateText) return;
    try {
      await navigator.clipboard.writeText(coordinateText.replace("\n", ", "));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const boundPoints = useMemo<[number, number][]>(() => {
    const pts: [number, number][] = [];
    if (recommended?.coordinates?.length) pts.push(...recommended.coordinates);
    alternatives.forEach((alt) => {
      if (alt.coordinates?.length) pts.push(...alt.coordinates);
    });
    trajectory?.observations.forEach((p) => pts.push([p.latitude, p.longitude]));
    trajectory?.predictions.forEach((p) => pts.push([p.latitude, p.longitude]));
    icebergs.forEach((ib) => pts.push([ib.latitude, ib.longitude]));
    if (startPoint) pts.push([startPoint.lat, startPoint.lon]);
    if (endPoint) pts.push([endPoint.lat, endPoint.lon]);
    if (vesselPos) pts.push(vesselPos);
    return pts;
  }, [recommended, alternatives, trajectory, icebergs, startPoint, endPoint, vesselPos]);

  return (
    <div className="relative w-full" style={{ height }}>
      <MapContainer
        center={savedView?.center ?? [-68, 80]}
        zoom={savedView?.zoom ?? 3.2}
        minZoom={2}
        maxZoom={8}
        className="h-full w-full"
        scrollWheelZoom
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <CoordinateInteraction
          onMove={handleMapMove}
          onLeave={handleMapLeave}
          onTap={handleMapTap}
        />
        <GraticuleLayer visible={graticuleVisible} />
        <RefreshMapLayer refreshToken={refreshToken} />
        <FocusPointLayer point={focusPoint} />
        <PersistMapViewLayer />

        {seaIce && <SeaIceOverlay seaIce={seaIce} />}

        {/* Selected-area boundary: dashed purple rectangle covering 55°S–75°S
         * with the same labels on the left and right edges.  Drawn from real
         * lat/lon (never screen coords), so it stays put while the user zooms
         * or pans the basemap.  Display-only decoration of the selected band. */}
        {seaIce && seaIce.lon.length > 1 && (
          <>
            {/* Dashed purple band boundary */}
            <Rectangle
              bounds={[
                [SELECTED_LAT_MIN, seaIce.lon[0]],
                [SELECTED_LAT_MAX, seaIce.lon[seaIce.lon.length - 1]],
              ]}
              pathOptions={{
                color: SELECTED_BAND_COLOR,
                weight: 2,
                dashArray: "9 7",
                fill: false,
              }}
            />
            {/* 55°S label — top edge, both left and right sides */}
            {[seaIce.lon[0], seaIce.lon[seaIce.lon.length - 1]].map((lon, i) => (
              <Marker
                key={`band55-${i}`}
                position={[SELECTED_LAT_MAX, lon]}
                icon={bandLabelIcon("55°S")}
                interactive={false}
              />
            ))}
            {/* 75°S label — bottom edge, both left and right sides */}
            {[seaIce.lon[0], seaIce.lon[seaIce.lon.length - 1]].map((lon, i) => (
              <Marker
                key={`band75-${i}`}
                position={[SELECTED_LAT_MIN, lon]}
                icon={bandLabelIcon("75°S")}
                interactive={false}
              />
            ))}
          </>
        )}

        {/* Recommended / current route — thick solid navy-blue line */}
        {recommended && recommended.coordinates.length > 0 && (
          <Polyline
            positions={recommended.coordinates}
            smoothFactor={1.5}
            pathOptions={{
              color: NAVY,
              weight: 5,
              opacity: 0.95,
              lineCap: "round",
              lineJoin: "round",
            }}
          />
        )}

        {/* The planner ends at a safe ocean approach cell when a station is
         * on land. Show that final station approach separately so the route
         * remains visibly connected without presenting land as navigable. */}
        {recommended && recommended.coordinates.length > 0 && endPoint && (
          <Polyline
            positions={[recommended.coordinates[recommended.coordinates.length - 1], [endPoint.lat, endPoint.lon]]}
            pathOptions={{
              color: "#f59e0b",
              weight: 3,
              opacity: 0.9,
              dashArray: "7 6",
              lineCap: "round",
            }}
          />
        )}

        {/* Alternative routes — thin dashed blue line */}
        {alternatives.map((alt, i) =>
          alt.coordinates.length > 0 ? (
            <Polyline
              key={i}
              positions={alt.coordinates}
              smoothFactor={1.5}
              pathOptions={{
                color: "#60a5fa",
                weight: 2,
                opacity: 0.85,
                dashArray: "8 6",
                lineCap: "round",
                lineJoin: "round",
              }}
            />
          ) : null,
        )}

        {trajectory && <TrajectoryLayer trajectory={trajectory} />}

        {/*
         * Dashboard display rule (only one change to the map): draw only the
         * icebergs that are at sea and within the coastal band nearest to the
         * Antarctic coast.  This is a view-level filter — it never touches the
         * count, the analytics, predictions, or the trajectory.
         * Fallback: a missing annotation keeps the iceberg visible (sea-only
         * data would otherwise go blank).
         */}
        {icebergs
          .filter(
            (ib) =>
              !ib.on_land &&
              ib.latitude >= SELECTED_LAT_MIN &&
              ib.latitude <= SELECTED_LAT_MAX
          )
          .map((ib) => (
          <CircleMarker
            key={ib.iceberg_id}
            center={[ib.latitude, ib.longitude]}
            radius={selectedIceberg === ib.iceberg_id ? 9 : 5}
            pathOptions={{
              color: selectedIceberg === ib.iceberg_id ? "#ef4444" : "#0ea5e9",
              fillColor: selectedIceberg === ib.iceberg_id ? "#ef4444" : "#38bdf8",
              fillOpacity: 0.9,
              weight: 2,
            }}
          >
            <Popup>
              <div className="text-xs">
                <strong>{ib.iceberg_id}</strong>
                <br />
                {ib.latitude.toFixed(3)}, {ib.longitude.toFixed(3)}
                {ib.length_km != null && (
                  <>
                    <br />
                    {ib.length_km}×{ib.width_km ?? "?"} km
                  </>
                )}
                {ib.demo && (
                  <>
                    <br />
                    <span className="text-amber-600">Demo</span>
                  </>
                )}
              </div>
            </Popup>
          </CircleMarker>
        ))}

        {startPoint && (
          <Marker position={[startPoint.lat, startPoint.lon]} icon={namedLabelIcon(`Departure · ${startPoint.name}`, "#10b981")}>
            <Popup>
              <div className="text-xs">
                <strong>Departure</strong>
                <br />
                {startPoint.name}
                <br />
                {startPoint.lat.toFixed(3)}, {startPoint.lon.toFixed(3)}
              </div>
            </Popup>
          </Marker>
        )}

        {endPoint && (
          <Marker position={[endPoint.lat, endPoint.lon]} icon={namedLabelIcon(`Destination · ${endPoint.name}`, "#0b3d6e")}>
            <Popup>
              <div className="text-xs">
                <strong>Destination</strong>
                <br />
                {endPoint.name}
                <br />
                {endPoint.lat.toFixed(3)}, {endPoint.lon.toFixed(3)}
              </div>
            </Popup>
          </Marker>
        )}

        {/* Clearly visible vessel marker */}
        {vesselPos && (
          <Marker
            position={vesselPos}
            icon={vesselIcon()}
            title={vesselLabel}
            zIndexOffset={1000}
          >
            <Popup>
              <div className="text-xs">
                <strong>{vesselLabel}</strong>
                <br />
                {vesselPos[0].toFixed(3)}, {vesselPos[1].toFixed(3)}
              </div>
            </Popup>
          </Marker>
        )}

        <FitBoundsLayer points={boundPoints} enabled={fitBounds} />
      </MapContainer>

      <div className="absolute right-3 top-3 z-[1000] flex items-start gap-2">
        <button
          type="button"
          className={`map-control-button ${graticuleVisible ? "map-control-button-active" : ""}`}
          onClick={() => setGraticuleVisible((visible) => !visible)}
          aria-pressed={graticuleVisible}
          title="Toggle latitude and longitude grid"
        >
          Grid
        </button>
      </div>

      {cursorPoint && cursorCoordinate && (
        <div
          className="map-coordinate-crosshair"
          style={{ left: cursorPoint.x, top: cursorPoint.y }}
          aria-hidden="true"
        />
      )}

      {cursorCoordinate && (
        <div className="map-coordinate-panel" role="status" aria-live="polite">
          <div className="map-coordinate-values">
            <span>Latitude: {formatCoordinate(cursorCoordinate.lat, "N", "S")}</span>
            <span>Longitude: {formatCoordinate(cursorCoordinate.lng, "E", "W")}</span>
          </div>
          <button
            type="button"
            className="map-copy-coordinate"
            onClick={copyCoordinates}
            title="Copy coordinates"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      )}

      {seaIce && showSeaIceLegend && (
        <SeaIceLegend />
      )}
    </div>
  );
}