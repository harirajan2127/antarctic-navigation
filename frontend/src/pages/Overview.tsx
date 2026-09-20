import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api";
import type {
  IcebergsListResponse,
  SeaIceForecastResponse,
  AnalyticsSummaryResponse,
} from "../types";
import { StatCard } from "../components/common/StatCard";
import { Card, CardHeader, CardBody } from "../components/common/Card";
import { LoadingState, ErrorState } from "../components/common/States";
import { useApp } from "../context/AppContext";

export function Overview() {
  const { toast, activeRoute } = useApp();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [icebergs, setIcebergs] = useState<IcebergsListResponse | null>(null);
  const [forecast, setForecast] = useState<SeaIceForecastResponse | null>(null);
  const [summary, setSummary] = useState<AnalyticsSummaryResponse | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [ib, fc, sm] = await Promise.all([
        api.icebergs(),
        api.seaIceForecast(24),
        api.analyticsSummary(),
      ]);
      setIcebergs(ib);
      setForecast(fc);
      setSummary(sm);

    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load overview data");
    } finally {
      setLoading(false);
    }
  }, []);

  const route = activeRoute;

  useEffect(() => {
    load();
    toast("Dashboard loaded with live Antarctic data", "info");
  }, [load, toast]);

  if (loading) return <LoadingState message="Loading overview..." />;
  if (error) return <ErrorState message={error} retry={load} />;

  return (
    <div className="space-y-5">
      <h1 className="text-lg font-bold text-navy-900">Overview</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          label="Total Icebergs"
          value={icebergs?.count ?? 0}
          sub={icebergs?.classification}
          accent="cyan"
        />
        <StatCard
          label="Sea-Ice Mean"
          value={forecast ? `${(forecast.mean_concentration * 100).toFixed(1)}%` : "—"}
          sub={`Horizon: ${forecast?.horizon_hours ?? 24}h`}
          accent="blue"
        />
        <StatCard
          label="Route Distance"
          value={route ? `${route.recommended.distance_km.toFixed(0)} km` : "—"}
          sub={route ? `${route.recommended.distance_nm.toFixed(0)} nm` : undefined}
          accent="green"
        />
        <StatCard
          label="Risk Level"
          value={route?.recommended.risk_level?.toUpperCase() ?? "—"}
          sub={route ? `${(route.recommended.risk_score ?? 0).toFixed(2)} score` : undefined}
          accent="amber"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader title="Datasets" subtitle={`${summary?.datasets?.length ?? 0} registered`} />
          <CardBody>
            <div className="space-y-2">
              {summary?.datasets?.map((d) => (
                <div key={d.dataset} className="flex items-center justify-between text-xs">
                  <span className="text-navy-700 font-medium">{d.dataset}</span>
                  <span className={`px-2 py-0.5 rounded-full font-medium ${d.demo ? "bg-amber-50 text-amber-600" : "bg-emerald-50 text-emerald-600"}`}>
                    {d.classification}
                  </span>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Model Status" subtitle={`${summary?.model_metrics_count ?? 0} metrics`} />
          <CardBody>
            <div className="space-y-2">
              {summary?.model_status && Object.values(summary.model_status).map((ms) => {
                const status = ms.status;
                const isReal = status === "trained_on_real_data" || status === "evaluated_on_real_data";
                const label =
                  {
                    trained_on_real_data: "Trained on Real Data",
                    evaluated_on_real_data: "Evaluated on Real Data",
                    model_available_source_unknown: "Model Available — Source Unknown",
                    demo_only: "Demo Only",
                    not_available: "Not Available",
                  }[status] ?? "Not Available";
                const chip = isReal
                  ? "bg-emerald-50 text-emerald-600"
                  : status === "demo_only"
                  ? "bg-amber-50 text-amber-600"
                  : "bg-slate-100 text-slate-500";
                return (
                  <div key={ms.model_name} className="flex items-center justify-between text-xs">
                    <span className="text-navy-700 font-medium">{ms.display_name}</span>
                    <span className={`px-2 py-0.5 rounded-full font-medium ${chip}`}>{label}</span>
                  </div>
                );
              })}
              {!summary?.model_status && summary?.real_model_available && Object.entries(summary.real_model_available).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between text-xs">
                  <span className="text-navy-700 font-medium capitalize">{k.replace("_", "-")}</span>
                  <span className={`px-2 py-0.5 rounded-full font-medium ${v ? "bg-emerald-50 text-emerald-600" : "bg-amber-50 text-amber-600"}`}>
                    {v ? "Real Model" : "Demo Only"}
                  </span>
                </div>
              ))}
              {summary?.model_status && Object.values(summary.model_status).map((ms) => {
                const k = ms.model_name;
                const m = ms.metrics ?? {};
                const acc = summary?.model_accuracy?.[k];
                const real = typeof m.mae === "number" || typeof m.mean_error_km === "number";
                if (!real) return null;
                return (
                  <div key={`acc-${k}`} className="flex items-center justify-between gap-2 text-xs">
                    <span className="text-navy-700 font-medium capitalize">{k.replace("_", "-")} accuracy</span>
                    <span className="text-[11px] text-emerald-600 text-right">
                      {k === "sea_ice"
                        ? `MAE ${(m.mae ?? acc?.mae)?.toFixed(4)} · RMSE ${(m.rmse ?? acc?.rmse)?.toFixed(4)} · R² ${(m.r2 ?? acc?.r2)?.toFixed(4)} (n=${m.test_samples ?? acc?.n_test_pairs})`
                        : `mean ${(m.mean_error_km ?? acc?.mean_km)?.toFixed(1)} km · median ${(m.median_error_km ?? acc?.median_km)?.toFixed(1)} km · ≤10km ${(m.within_10km_percent ?? acc?.within_10km_pct)?.toFixed(2)}% (n=${m.test_samples ?? acc?.n_test_pairs})`}
                    </span>
                  </div>
                );
              })}
              {!summary?.model_status && summary?.model_accuracy && Object.entries(summary.model_accuracy).map(([k, acc]) => (
                <div key={`acc-${k}`} className="flex items-center justify-between gap-2 text-xs">
                  <span className="text-navy-700 font-medium capitalize">{k.replace("_", "-")} accuracy</span>
                  {acc.available ? (
                    <span className="text-[11px] text-emerald-600 text-right">
                      {k === "sea_ice"
                        ? `MAE ${acc.mae?.toFixed(4)} · RMSE ${acc.rmse?.toFixed(4)} · R² ${acc.r2?.toFixed(4)} (n=${acc.n_test_pairs})`
                        : `mean ${acc.mean_km?.toFixed(1)} km · median ${acc.median_km?.toFixed(1)} km · ≤10km ${acc.within_10km_pct}% (n=${acc.n_test_pairs})`}
                    </span>
                  ) : (
                    <span className="text-[11px] text-amber-600 text-right">
                      {acc.reason ?? "Accuracy Not Available — Ground Truth Missing"}
                    </span>
                  )}
                </div>
              ))}
            </div>
            {summary?.warnings?.map((w, i) => (
              <p key={i} className="text-[11px] text-amber-600 mt-2">{w}</p>
            ))}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Navigation Evaluation" subtitle="Measured real-data checks" />
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(["two_hour_recalculation", "land_mask"] as const).map((key) => {
              const check = summary?.evaluation_checks?.[key];
              const title = key === "two_hour_recalculation" ? "2-Hour Recalculation" : "Land Mask";
              const status = check?.status ?? "UNAVAILABLE";
              const measured = typeof check?.accuracy_percent === "number";
              return (
                <div key={key} className="border border-slate-200 rounded-lg p-3 space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-bold text-navy-900">{title}</p>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      status === "VERIFIED" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
                    }`}>
                      {status === "INSUFFICIENT_DATA" ? "Insufficient data" : status === "UNAVAILABLE" ? "Unavailable" : status}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <span className="text-navy-500">Actual score</span>
                    <span className="text-right font-semibold text-navy-900">{measured ? `${check.accuracy_percent?.toFixed(2)}%` : "N/A"}</span>
                    <span className="text-navy-500">Target score</span>
                    <span className="text-right text-navy-700">{check?.target_percent != null ? `${check.target_percent}%` : "N/A"}</span>
                    {key === "two_hour_recalculation" ? (
                      <>
                        <span className="text-navy-500">Evaluated predictions</span>
                        <span className="text-right text-navy-700">{check?.number_of_valid_2h_evaluations ?? "N/A"}</span>
                        <span className="text-navy-500">Average error</span>
                        <span className="text-right text-navy-700">{check?.average_distance_error_km != null ? `${check.average_distance_error_km.toFixed(2)} km` : "N/A"}</span>
                      </>
                    ) : (
                      <>
                        <span className="text-navy-500">Ocean-valid points</span>
                        <span className="text-right text-navy-700">{check?.ocean_points ?? "N/A"}</span>
                        <span className="text-navy-500">Land violations</span>
                        <span className="text-right text-navy-700">{check?.land_points ?? "N/A"}</span>
                      </>
                    )}
                  </div>
                  {!measured && check?.message && <p className="text-[11px] text-amber-700">{check.message}</p>}
                </div>
              );
            })}
          </div>
        </CardBody>
      </Card>

      {summary?.accuracy_summary && (
        <Card>
          <CardHeader title="Accuracy Summary" subtitle="Validated weighted calculation" />
          <CardBody>
            <div className="flex flex-wrap items-end gap-x-8 gap-y-2 mb-4">
              <div>
                <p className="text-[11px] text-navy-500">Total weighted accuracy</p>
                <p className="text-xl font-bold text-navy-900">
                  {summary.accuracy_summary.score != null
                    ? `${summary.accuracy_summary.provisional ? "PROVISIONAL " : ""}${summary.accuracy_summary.score.toFixed(2)}%`
                    : "N/A"}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-navy-500">Available weight coverage</p>
                <p className="text-sm font-semibold text-navy-800">{summary.accuracy_summary.available_weight_coverage_percent.toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-[11px] text-navy-500">Overall grade</p>
                <p className="text-sm font-semibold text-navy-800">{summary.accuracy_summary.grade}</p>
              </div>
            </div>
            {summary.accuracy_summary.missing_components.length > 0 && (
              <p className="text-[11px] text-amber-700 mb-3">
                Missing component: {summary.accuracy_summary.missing_components.join(", ")}
              </p>
            )}
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead className="text-left text-navy-500 border-b border-slate-200">
                  <tr><th className="py-2">Component</th><th>Accuracy</th><th>Weight</th><th>Contribution</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {summary.accuracy_summary.breakdown.map((row) => (
                    <tr key={row.component} className="border-b border-slate-100">
                      <td className="py-2 font-medium text-navy-800">{row.label}</td>
                      <td>{row.accuracy != null ? `${row.accuracy.toFixed(2)}%` : "N/A"}</td>
                      <td>{(row.weight * 100).toFixed(1)}%</td>
                      <td>{row.weighted_contribution != null ? row.weighted_contribution.toFixed(2) : "N/A"}</td>
                      <td className={row.status === "Valid" ? "text-emerald-700" : "text-amber-700"}>{row.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardBody>
        </Card>
      )}

      {route && (
        <Card>
          <CardHeader
            title="Active Route"
            subtitle={`${route.start_latitude.toFixed(1)}, ${route.start_longitude.toFixed(1)} → ${route.destination_name ?? `${route.destination_latitude.toFixed(1)}, ${route.destination_longitude.toFixed(1)}`}`}
          />
          <CardBody>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
              {[
                { label: "Distance", value: `${route.recommended.distance_km.toFixed(0)} km` },
                { label: "Travel Time", value: `${route.recommended.travel_time_hours.toFixed(1)} h` },
                { label: "Fuel", value: route.recommended.fuel_tons ? `${route.recommended.fuel_tons.toFixed(1)} t` : "—" },
                { label: "Waypoints", value: route.recommended.waypoints.length },
                { label: "Alternatives", value: route.alternatives.length },
              ].map((item) => (
                <div key={item.label}>
                  <p className="text-[11px] text-navy-400">{item.label}</p>
                  <p className="text-sm font-bold text-navy-900">{item.value}</p>
                </div>
              ))}
            </div>
            {route.ocean_approach_distance_km != null && (
              <p className="mt-3 text-[11px] text-navy-400">
                {route.destination_name ?? "Destination"} is on land; routed to the nearest
                ocean cell {route.ocean_approach_distance_km.toFixed(0)} km away
                ({route.destination_latitude.toFixed(2)}, {route.destination_longitude.toFixed(2)}).
              </p>
            )}
            {route.recommended.warnings.length > 0 && (
              <div className="mt-3 p-2 bg-amber-50 rounded-lg border border-amber-100">
                {route.recommended.warnings.map((w, i) => (
                  <p key={i} className="text-[11px] text-amber-700">{w}</p>
                ))}
              </div>
            )}
          </CardBody>
        </Card>
      )}
    </div>
  );
}