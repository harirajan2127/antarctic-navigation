import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import { haversineKm, kmToNm } from "../utils/haversine";
import type {
  IcebergsListResponse,
  IcebergTrajectoryResponse,
  IcebergDetailResponse,
  IcebergDistanceResponse,
} from "../types";
import { Card, CardHeader, CardBody } from "../components/common/Card";
import { LoadingState, ErrorState } from "../components/common/States";
import { useApp } from "../context/AppContext";
import { Antarctic3DGlobe } from "../components/maps/Antarctic3DGlobe";

const VESSEL_POS: [number, number] = [-68.0, 147.0];

export function IcebergTrackingPage() {
  const { toast } = useApp();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [icebergs, setIcebergs] = useState<IcebergsListResponse | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "near" | "large">("all");
  const [selected, setSelected] = useState<IcebergDetailResponse | null>(null);
  const [trajectory, setTrajectory] = useState<IcebergTrajectoryResponse | null>(null);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  const [distance, setDistance] = useState<IcebergDistanceResponse | null>(null);
  const [fromId, setFromId] = useState("");
  const [toId, setToId] = useState("");

  const loadIcebergs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ib = await api.icebergs();
      setIcebergs(ib);
      const first = ib.icebergs[0];
      if (first) {
        setFromId(first.iceberg_id);
        const second = ib.icebergs[1];
        if (second) setToId(second.iceberg_id);
      }
      toast("Icebergs loaded", "success");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load icebergs");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadIcebergs();
  }, [loadIcebergs]);

  const selectIceberg = useCallback(async (id: string) => {
    try {
      const [det, traj, dist] = await Promise.all([
        api.icebergDetail(id),
        api.icebergTrajectory(id),
        fromId && id !== fromId ? api.icebergDistance(fromId, id) : Promise.resolve(null),
      ]);
      setSelected(det);
      setTrajectory(traj);
      setTrajectoryError(null);
      setDistance(dist);
    } catch (e) {
      setTrajectory(null);
      setTrajectoryError(e instanceof Error ? e.message : "Iceberg trajectory unavailable.");
      toast(e instanceof Error ? e.message : "Failed to load iceberg detail", "error");
    }
  }, [fromId, toast]);

  useEffect(() => {
    if (icebergs?.icebergs[0] && !selected) {
      selectIceberg(icebergs.icebergs[0].iceberg_id);
    }
  }, [icebergs, selected, selectIceberg]);

  const filtered = useMemo(() => {
    if (!icebergs) return [];
    let list = icebergs.icebergs;
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((i) => i.iceberg_id.toLowerCase().includes(q));
    }
    if (filter === "near") {
      list = list
        .map((i) => ({ i, d: haversineKm(i.latitude, i.longitude, VESSEL_POS[0], VESSEL_POS[1]) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 10)
        .map((x) => x.i);
    } else if (filter === "large") {
      list = list
        .filter((i) => (i.length_km ?? 0) > 0)
        .sort((a, b) => (b.length_km ?? 0) - (a.length_km ?? 0))
        .slice(0, 10);
    }
    return list;
  }, [icebergs, search, filter]);

  const calculateDistance = useCallback(async () => {
    if (!fromId || !toId) {
      toast("Select both icebergs", "error");
      return;
    }
    try {
      const d = await api.icebergDistance(fromId, toId);
      setDistance(d);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Failed to calculate distance", "error");
    }
  }, [fromId, toId, toast]);

  if (loading) return <LoadingState message="Loading iceberg tracking..." />;
  if (error) return <ErrorState message={error} retry={loadIcebergs} />;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-bold text-navy-900">Iceberg Tracking</h1>
          <p className="text-xs text-navy-400 mt-0.5">
            {icebergs?.demo ? "Synthetic demo data" : "Real observations"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search iceberg ID"
            className="text-xs border border-slate-200 rounded-lg px-3 py-1.5 bg-white text-navy-700 w-40"
          />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as typeof filter)}
            className="text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-navy-700"
          >
            <option value="all">All</option>
            <option value="near">Nearest 10</option>
            <option value="large">Largest 10</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1 space-y-4">
          <Card>
            <CardHeader title="Iceberg List" subtitle={`${filtered.length} shown / ${icebergs?.count ?? 0}`} />
            <CardBody className="max-h-[360px] overflow-y-auto">
              <div className="space-y-1">
                {filtered.length === 0 && (
                  <p className="text-xs text-navy-400 py-4 text-center">No icebergs found</p>
                )}
                {filtered.map((ib) => (
                  <button
                    key={ib.iceberg_id}
                    onClick={() => selectIceberg(ib.iceberg_id)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors border ${
                      selected?.iceberg_id === ib.iceberg_id
                        ? "bg-blue-50 border-blue-200"
                        : "border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-navy-800">{ib.iceberg_id}</span>
                      <span className="text-navy-400">
                        {haversineKm(ib.latitude, ib.longitude, VESSEL_POS[0], VESSEL_POS[1]).toFixed(0)} km
                      </span>
                    </div>
                    <div className="text-[11px] text-navy-400 mt-0.5">
                      {ib.latitude.toFixed(3)}, {ib.longitude.toFixed(3)}
                    </div>
                  </button>
                ))}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Distance Between Icebergs" />
            <CardBody>
              <div className="space-y-2">
                <select
                  value={fromId}
                  onChange={(e) => setFromId(e.target.value)}
                  className="w-full text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-navy-700"
                >
                  {icebergs?.icebergs.map((i) => (
                    <option key={i.iceberg_id} value={i.iceberg_id}>{i.iceberg_id}</option>
                  ))}
                </select>
                <select
                  value={toId}
                  onChange={(e) => setToId(e.target.value)}
                  className="w-full text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-navy-700"
                >
                  {icebergs?.icebergs.map((i) => (
                    <option key={i.iceberg_id} value={i.iceberg_id}>{i.iceberg_id}</option>
                  ))}
                </select>
                <button
                  onClick={calculateDistance}
                  className="w-full px-3 py-2 text-xs font-semibold bg-accent-blue text-white rounded-lg hover:bg-accent-blue-dark transition-colors"
                >
                  Calculate
                </button>
                {distance && (
                  <div className="text-center">
                    <p className="text-2xl font-bold text-navy-900">
                      {distance.distance_km.toFixed(0)} km
                    </p>
                    <p className="text-[11px] text-navy-400">
                      {distance.distance_nm.toFixed(0)} nm
                    </p>
                  </div>
                )}
              </div>
            </CardBody>
          </Card>
        </div>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Map"
            subtitle={selected ? `Selected: ${selected.iceberg_id}` : "Select an iceberg"}
          />
          <CardBody>
            <div className="h-[480px] rounded-lg overflow-hidden">
              <Antarctic3DGlobe
                icebergs={icebergs?.icebergs ?? []}
                selectedIceberg={selected?.iceberg_id}
                trajectory={trajectory}
                vesselPos={VESSEL_POS}
                onIcebergSelect={selectIceberg}
                height="480px"
              />
            </div>
            {trajectory && (
              <div className="flex items-center gap-4 mt-2 text-[11px] text-navy-500">
                <span className="flex items-center gap-1">
                  <span className="w-3 h-1.5 bg-indigo-500 inline-block rounded" /> Historical
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-1.5 bg-amber-500 inline-block rounded" /> Predicted
                </span>
                <span>
                  {trajectory.count_observations} obs · {trajectory.count_predictions} pred
                </span>
              </div>
            )}
            {trajectoryError && (
              <div className="mt-3 flex items-center justify-between gap-3 text-xs text-red-700">
                <span>Iceberg trajectory unavailable. Reason: {trajectoryError}</span>
                {selected && (
                  <button
                    type="button"
                    onClick={() => selectIceberg(selected.iceberg_id)}
                    className="px-3 py-1.5 rounded-lg bg-red-50 hover:bg-red-100 font-medium"
                  >
                    Retry
                  </button>
                )}
              </div>
            )}
          </CardBody>
        </Card>
      </div>

      {selected && (
        <Card>
          <CardHeader title={`Iceberg ${selected.iceberg_id}`} subtitle={selected.classification} />
          <CardBody>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              {[
                { label: "Latitude", value: selected.latitude.toFixed(4) },
                { label: "Longitude", value: selected.longitude.toFixed(4) },
                { label: "Length", value: selected.length_km ? `${selected.length_km} km` : "—" },
                { label: "Observations", value: selected.observation_count },
                { label: "Distance from Vessel", value: `${kmToNm(haversineKm(selected.latitude, selected.longitude, VESSEL_POS[0], VESSEL_POS[1])).toFixed(1)} nm` },
              ].map((s) => (
                <div key={s.label}>
                  <p className="text-[11px] text-navy-400">{s.label}</p>
                  <p className="text-sm font-bold text-navy-900">{s.value}</p>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}