import { useMemo, useState } from "react";
import {
  BellAlertIcon,
  CheckIcon,
  ChevronDownIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  MagnifyingGlassIcon,
  ShieldExclamationIcon,
  TrashIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { Card, CardBody } from "../components/common/Card";
import { useApp } from "../context/AppContext";
import type { AlertRecord, AlertSeverity, AlertType } from "../types";

const severityStyles: Record<AlertSeverity, { label: string; classes: string; icon: typeof InformationCircleIcon }> = {
  info: { label: "Info", classes: "bg-blue-50 text-blue-700 border-blue-200", icon: InformationCircleIcon },
  low: { label: "Low", classes: "bg-slate-50 text-slate-600 border-slate-200", icon: InformationCircleIcon },
  medium: { label: "Medium", classes: "bg-yellow-50 text-yellow-700 border-yellow-200", icon: ExclamationTriangleIcon },
  warning: { label: "Warning", classes: "bg-amber-50 text-amber-700 border-amber-200", icon: ExclamationTriangleIcon },
  high: { label: "High", classes: "bg-orange-50 text-orange-700 border-orange-200", icon: ShieldExclamationIcon },
  critical: { label: "Critical", classes: "bg-red-50 text-red-700 border-red-200", icon: BellAlertIcon },
};

const typeLabels: Record<AlertType, string> = {
  route: "Route",
  "sea-ice": "Sea Ice",
  iceberg: "Iceberg",
  weather: "Weather",
  system: "System",
};

type Filter = "all" | "unread" | AlertSeverity | AlertType;

function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function routeLabel(alert: AlertRecord): string | null {
  if (alert.selectedRouteLabel) return alert.selectedRouteLabel;
  const route = alert.newRoute ?? alert.previousRoute;
  if (!route) return null;
  return `${route.start_latitude.toFixed(2)}, ${route.start_longitude.toFixed(2)} to ${route.destination_latitude.toFixed(2)}, ${route.destination_longitude.toFixed(2)}`;
}

function AlertDetails({ alert, onFocus }: { alert: AlertRecord; onFocus: (alert: AlertRecord) => void }) {
  const previous = alert.previousRoute?.recommended;
  const next = alert.newRoute?.recommended;
  return (
    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 border-t border-slate-100 pt-3 text-xs">
      {alert.reason && <div className="sm:col-span-2"><span className="text-navy-400">Reason</span><p className="text-navy-700 mt-0.5">{alert.reason}</p></div>}
      {alert.selectedRouteLabel && <div><span className="text-navy-400">Selected route</span><p className="font-semibold text-navy-800">{alert.selectedRouteLabel}</p></div>}
      {alert.icebergId && <div><span className="text-navy-400">Iceberg ID</span><p className="font-semibold text-navy-800">{alert.icebergId}</p></div>}
      {alert.latitude != null && alert.distanceKm != null && <div><span className="text-navy-400">Distance from route</span><p className="text-navy-700">{alert.distanceKm.toFixed(1)} km · {alert.distanceNm?.toFixed(1)} NM</p></div>}
      {alert.latitude != null && <div><span className="text-navy-400">Location</span><p className="text-navy-700">{formatCoordinate(alert.latitude, true)}, {formatCoordinate(alert.longitude ?? 0, false)}</p></div>}
      {alert.recommendedAction && <div className="sm:col-span-2"><span className="text-navy-400">Recommended action</span><p className="text-navy-700 mt-0.5">{alert.recommendedAction}</p></div>}
      {alert.latitude != null && alert.longitude != null && <button type="button" onClick={() => onFocus(alert)} className="sm:col-span-2 text-left text-xs font-semibold text-accent-blue hover:text-blue-800">View location on navigation map</button>}
      {previous && <div><span className="text-navy-400">Previous route</span><p className="text-navy-700">{previous.distance_nm.toFixed(0)} NM · {previous.travel_time_hours.toFixed(1)} h · {previous.risk_level ?? "unknown"}</p></div>}
      {next && <div><span className="text-navy-400">New route</span><p className="text-navy-700">{next.distance_nm.toFixed(0)} NM · {next.travel_time_hours.toFixed(1)} h · {next.risk_level ?? "unknown"}</p></div>}
      {alert.metadata && Object.entries(alert.metadata).map(([key, value]) => (
        <div key={key}><span className="text-navy-400">{key.replace(/([A-Z])/g, " $1")}</span><p className="text-navy-700">{typeof value === "number" ? value.toFixed(1) : value ?? "-"}</p></div>
      ))}
    </div>
  );
}

function formatCoordinate(value: number, latitude: boolean): string {
  return `${Math.abs(value).toFixed(4)}°${latitude ? (value >= 0 ? " N" : " S") : value >= 0 ? " E" : " W"}`;
}

export function AlertMessagePage() {
  const { alerts, markAlertRead, markAllAlertsRead, deleteAlert, clearAlerts, setPage, setNavigation } = useApp();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return alerts.filter((alert) => {
      const matchesFilter = filter === "all"
        || (filter === "unread" && !alert.read)
        || alert.severity === filter
        || alert.type === filter;
      const text = [alert.title, alert.message, alert.reason, alert.icebergId, routeLabel(alert), alert.latitude, alert.longitude]
        .filter(Boolean).join(" ").toLowerCase();
      return matchesFilter && (!query || text.includes(query));
    });
  }, [alerts, filter, search]);

  const unread = alerts.filter((alert) => !alert.read).length;
  const critical = alerts.filter((alert) => alert.severity === "critical").length;
  const warnings = alerts.filter((alert) => alert.severity === "warning" || alert.severity === "high" || alert.severity === "medium").length;
  const lastUpdated = alerts.length ? new Date(Math.max(...alerts.map((alert) => Date.parse(alert.timestamp)))).toLocaleTimeString() : "-";
  const focusAlert = (alert: AlertRecord) => {
    if (alert.latitude == null || alert.longitude == null) return;
    setNavigation({ alertFocus: [alert.latitude, alert.longitude] });
    setPage("planner");
  };

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div><h1 className="text-lg font-bold text-navy-900">Alert Message</h1><p className="text-xs text-navy-400 mt-0.5">Navigation alerts and system updates</p></div>
        <div className="flex gap-2">
          <button type="button" onClick={markAllAlertsRead} disabled={!unread} className="px-3 py-1.5 text-xs font-semibold border border-slate-200 rounded-lg text-navy-600 hover:bg-slate-50 disabled:opacity-40">Mark All as Read</button>
          <button type="button" onClick={() => { if (alerts.length && window.confirm("Clear all alerts? This only removes alert records.")) clearAlerts(); }} disabled={!alerts.length} className="px-3 py-1.5 text-xs font-semibold border border-red-200 rounded-lg text-red-600 hover:bg-red-50 disabled:opacity-40">Clear All</button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[{ label: "Total Alerts", value: alerts.length, color: "text-navy-900" }, { label: "Unread", value: unread, color: "text-blue-700" }, { label: "Critical", value: critical, color: "text-red-700" }, { label: "Warnings", value: warnings, color: "text-amber-700" }].map((item) => (
          <Card key={item.label}><CardBody className="pt-4"><p className="text-xs text-navy-400">{item.label}</p><p className={`text-2xl font-bold mt-1 ${item.color}`}>{item.value}</p></CardBody></Card>
        ))}
      </div>

      <Card>
        <CardBody className="pt-4">
          <div className="flex flex-col lg:flex-row gap-2">
            <div className="relative flex-1"><MagnifyingGlassIcon className="absolute left-3 top-2 w-4 h-4 text-navy-300" /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search alerts" className="w-full text-xs border border-slate-200 rounded-lg pl-9 pr-3 py-2 bg-white text-navy-700" /></div>
            <div className="flex gap-1.5 flex-wrap">{(["all", "unread", "critical", "high", "medium", "low", "info", "route", "sea-ice", "iceberg", "weather", "system"] as Filter[]).map((value) => <button key={value} type="button" onClick={() => setFilter(value)} className={`px-2.5 py-1.5 rounded-lg text-[11px] font-semibold border ${filter === value ? "bg-accent-blue/10 text-accent-blue border-accent-blue/30" : "border-slate-200 text-navy-500 hover:bg-slate-50"}`}>{value === "sea-ice" ? "Sea Ice" : value[0].toUpperCase() + value.slice(1)}</button>)}</div>
          </div>
        </CardBody>
      </Card>

      <p className="text-[11px] text-navy-400 text-right">Last Updated: {lastUpdated}</p>
      {filtered.length === 0 ? (
        <Card><CardBody className="py-12 text-center"><CheckIcon className="w-10 h-10 mx-auto text-emerald-500" /><h2 className="text-sm font-semibold text-navy-800 mt-3">All Clear</h2><p className="text-xs text-navy-400 mt-1">No new navigation alerts or system notifications.</p></CardBody></Card>
      ) : (
        <div className="space-y-3">{filtered.map((alert) => {
          const severity = severityStyles[alert.severity];
          const Icon = severity.icon;
          const isExpanded = expanded === alert.id;
          return <Card key={alert.id} className={!alert.read ? "border-l-4 border-l-accent-blue" : ""}>
            <CardBody className="pt-4">
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg border ${severity.classes}`}><Icon className="w-5 h-5" /></div>
                <div className="min-w-0 flex-1"><div className="flex items-start justify-between gap-2"><div><h2 className="text-sm font-semibold text-navy-900">{alert.title}</h2><p className="text-xs text-navy-500 mt-0.5">{alert.message}</p></div><span className={`shrink-0 px-2 py-0.5 rounded-full border text-[10px] font-semibold ${severity.classes}`}>{severity.label}</span></div>
                  <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[11px] text-navy-400"><span>{typeLabels[alert.type]}</span><span>{formatTime(alert.timestamp)}</span>{routeLabel(alert) && <span className="truncate">{routeLabel(alert)}</span>}{!alert.read && <span className="font-semibold text-accent-blue">Unread</span>}</div>
                  {isExpanded && <AlertDetails alert={alert} onFocus={focusAlert} />}
                </div>
                <div className="flex items-center gap-1"><button type="button" onClick={() => setExpanded(isExpanded ? null : alert.id)} className="p-1.5 text-navy-400 hover:text-navy-700" title="View details" aria-label="View details"><ChevronDownIcon className={`w-4 h-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} /></button><button type="button" onClick={() => markAlertRead(alert.id, !alert.read)} className="p-1.5 text-navy-400 hover:text-navy-700" title={alert.read ? "Mark as unread" : "Mark as read"} aria-label={alert.read ? "Mark as unread" : "Mark as read"}>{alert.read ? <XMarkIcon className="w-4 h-4" /> : <CheckIcon className="w-4 h-4" />}</button><button type="button" onClick={() => deleteAlert(alert.id)} className="p-1.5 text-navy-400 hover:text-red-600" title="Delete alert" aria-label="Delete alert"><TrashIcon className="w-4 h-4" /></button></div>
              </div>
            </CardBody>
          </Card>;
        })}</div>
      )}
    </div>
  );
}
