import { useEffect, useState } from "react";
import type { GridData } from "../../components/map/AntarcticMap";

export interface LandMask {
  step_deg: number;
  lat_start: number;
  lat_end: number;
  n_lat: number;
  lon_start: number;
  lon_end: number;
  n_lon: number;
  land: number[];
  ocean_cells: number;
  land_cells: number;
  total_cells: number;
  generated_at: string;
  source: string;
}

export type LandMaskState =
  | { status: "loading" }
  | { status: "ready"; mask: LandMask }
  | { status: "error"; message: string };

const LAND_MASK_URL = `${import.meta.env.BASE_URL ?? ""}data/land_mask.json`;

/** Says whether a (lat, lon) cell lies on land inside the mask's coverage.
 *  Cells outside the mask extent return `false` (treated as ocean, matching
 *  the backend land mask which only covers the coastal region). */
export function isLandCell(mask: LandMask, lat: number, lon: number): boolean {
  if (lat < mask.lat_start || lat > mask.lat_end) return false;
  if (lon < mask.lon_start || lon > mask.lon_end) return false;
  const c = Math.round((lon - mask.lon_start) / mask.step_deg);
  const r = Math.round((mask.lat_end - lat) / mask.step_deg);
  if (r < 0 || r >= mask.n_lat || c < 0 || c >= mask.n_lon) return false;
  return mask.land[r * mask.n_lon + c] === 1;
}

/** Fetches the real-data-derived coastal land mask shipped with the frontend. */
export function useLandMask(): LandMaskState {
  const [state, setState] = useState<LandMaskState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(LAND_MASK_URL);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const mask = (await res.json()) as LandMask;
        if (!Array.isArray(mask.land) || !mask.n_lat || !mask.n_lon) {
          throw new Error("malformed land mask");
        }
        if (!cancelled) setState({ status: "ready", mask });
      } catch (e) {
        if (!cancelled) {
          setState({
            status: "error",
            message: e instanceof Error ? e.message : "failed to load",
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}

/** Down-samples a sea-ice grid into `point{lat,lng,concentration}` objects,
 *  keeping only ocean cells whose concentration is visible (> 0).  An even
 *  row/col stride keeps the total under `maxPoints` for 3D-globe performance
 *  while preserving the full-region pattern (metrics still use the complete
 *  dataset on the API side, never these sampled points). */
export function buildSeaIcePoints(
  seaIce: GridData,
  isLand: (lat: number, lon: number) => boolean,
  maxPoints = 14000,
): { lat: number; lng: number; concentration: number }[] {
  const nLat = seaIce.lat.length;
  const nLon = seaIce.lon.length;
  if (!nLat || !nLon) return [];

  let densePoints = 0;
  for (let r = 0; r < nLat; r++) {
    const row = seaIce.concentration[r];
    if (!row) continue;
    for (let c = 0; c < nLon; c++) {
      const v = row[c];
      if (!v || v <= 0) continue;
      if (!isLand(seaIce.lat[r], seaIce.lon[c])) densePoints++;
    }
  }

  const stride = densePoints <= maxPoints ? 1 : Math.ceil(Math.sqrt(densePoints / maxPoints));

  const out: { lat: number; lng: number; concentration: number }[] = [];
  for (let r = 0; r < nLat; r += stride) {
    const row = seaIce.concentration[r];
    if (!row) continue;
    for (let c = 0; c < nLon; c += stride) {
      const v = row[c];
      if (!v || v <= 0) continue;
      const lat = seaIce.lat[r];
      const lon = seaIce.lon[c];
      if (isLand(lat, lon)) continue;
      out.push({ lat, lng: lon, concentration: v });
    }
  }
  return out;
}