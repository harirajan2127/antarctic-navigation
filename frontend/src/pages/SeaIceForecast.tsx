import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import type { SeaIceCurrentResponse, SeaIceForecastResponse } from "../types";
import { Card, CardHeader, CardBody } from "../components/common/Card";
import { LoadingState, ErrorState } from "../components/common/States";
import { useApp } from "../context/AppContext";
import { Antarctic3DGlobe } from "../components/maps/Antarctic3DGlobe";
import { seaIceLegendStops } from "../utils/colormap";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const HORIZONS = [6, 12, 24, 48, 72, 120, 168];

export function SeaIceForecastPage() {
  const { toast } = useApp();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<SeaIceCurrentResponse | null>(null);
  const [forecast, setForecast] = useState<SeaIceForecastResponse | null>(null);
  const [horizon, setHorizon] = useState(24);
  const [multiHorizon, setMultiHorizon] = useState<
    { horizon: number; mean: number; max: number; coverage: number }[]
  >([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [cur, fc] = await Promise.all([
        api.seaIceCurrent(),
        api.seaIceForecast(horizon),
      ]);
      setCurrent(cur);
      setForecast(fc);

      const multi = await Promise.all(
        HORIZONS.map(async (h) => {
          const f = await api.seaIceForecast(h);
          return {
            horizon: h,
            mean: +(f.mean_concentration * 100).toFixed(2),
            max: +(f.max_concentration * 100).toFixed(2),
            coverage: +f.coverage_pct.toFixed(1),
          };
        }),
      );
      setMultiHorizon(multi);
      toast("Sea-ice data loaded", "success");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load sea-ice data");
    } finally {
      setLoading(false);
    }
  }, [horizon, toast]);

  useEffect(() => {
    load();
  }, [load]);

  const legendStops = useMemo(() => seaIceLegendStops(), []);

  if (loading) return <LoadingState message="Loading sea-ice forecast..." />;
  if (error) return <ErrorState message={error} retry={load} />;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-bold text-navy-900">Sea-Ice Forecast</h1>
          <p className="text-xs text-navy-400 mt-0.5">
            {forecast?.demo ? "Synthetic demo data — not real forecast" : "Real data"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-navy-500">Horizon:</label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-navy-700"
          >
            {HORIZONS.map((h) => (
              <option key={h} value={h}>
                {h}h
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        {[
          { label: "Mean Concentration", value: forecast ? `${(forecast.mean_concentration * 100).toFixed(1)}%` : "—" },
          { label: "Max Concentration", value: forecast ? `${(forecast.max_concentration * 100).toFixed(1)}%` : "—" },
          { label: "Coverage", value: forecast ? `${forecast.coverage_pct.toFixed(1)}%` : "—" },
          { label: "Model", value: forecast?.model ?? "—" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-xl border border-slate-200 shadow-card p-3 text-center">
            <p className="text-[11px] text-navy-400">{s.label}</p>
            <p className="text-base font-bold text-navy-900 mt-0.5">{s.value}</p>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-slate-200 shadow-card overflow-hidden">
        <div className="px-5 pt-4 pb-2">
          <h3 className="text-sm font-semibold text-navy-900">Concentration Map</h3>
          <p className="text-[11px] text-navy-400">
            Forecast valid: {forecast?.forecast_time ? new Date(forecast.forecast_time).toLocaleString() : "—"}
          </p>
        </div>
        <div className="px-5 pb-2">
          <div className="flex items-center gap-2 mb-1">
            <div className="flex items-center gap-1">
              {legendStops.map((s, i) => (
                <div key={i} className="flex items-center gap-0.5">
                  <div className="w-6 h-3 rounded-sm" style={{ backgroundColor: s.color }} />
                  <span className="text-[10px] text-navy-500">{s.pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="h-[400px]">
          <Antarctic3DGlobe
            seaIce={forecast ?? current}
            seaIceTimestamp={forecast?.forecast_time ?? current?.timestamp ?? null}
            seaIceHorizon={horizon}
            height="400px"
          />
        </div>
      </div>

      {multiHorizon.length > 0 && (
        <Card>
          <CardHeader title="Multi-Horizon Forecast" subtitle="Mean concentration by forecast horizon" />
          <CardBody>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={multiHorizon}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="horizon" tick={{ fontSize: 11 }} label={{ value: "Hours", position: "bottom", fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(v: number) => [`${v}%`]}
                  />
                  <Line type="monotone" dataKey="mean" stroke="#2563eb" strokeWidth={2} name="Mean %" />
                  <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={2} name="Max %" />
                  <Line type="monotone" dataKey="coverage" stroke="#10b981" strokeWidth={2} name="Coverage %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </Card>
      )}

      {forecast?.skill_note && (
        <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg text-xs text-blue-700">
          {forecast.skill_note}
        </div>
      )}
    </div>
  );
}