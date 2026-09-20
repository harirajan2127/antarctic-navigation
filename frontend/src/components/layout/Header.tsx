import { useEffect, useState } from "react";
import { useApp } from "../../context/AppContext";
import { api } from "../../services/api";
import { Bars3Icon, BellAlertIcon, ChevronRightIcon, SignalIcon, WifiIcon } from "@heroicons/react/24/outline";

export function Header() {
  const { toggleSidebar, alerts, unreadAlertCount, markAlertRead, setPage } = useApp();
  const [backendStatus, setBackendStatus] = useState<"ok" | "error" | "loading">("loading");

  useEffect(() => {
    api.health().then(() => setBackendStatus("ok")).catch(() => setBackendStatus("error"));
  }, []);

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center px-4 lg:px-6 shrink-0">
      <button
        onClick={toggleSidebar}
        className="lg:hidden mr-3 text-navy-500 hover:text-navy-800"
      >
        <Bars3Icon className="w-5 h-5" />
      </button>

      <h2 className="text-sm font-medium text-navy-700 hidden sm:block">
        Antarctic Navigation Decision Support System
      </h2>

      <div className="ml-auto flex items-center gap-3">
        <div className="relative group">
          <button
            type="button"
            onClick={() => setPage("alerts")}
            className="relative p-2 text-navy-500 hover:text-accent-blue hover:bg-blue-50 rounded-lg"
            title="Open alerts"
            aria-label={`Open alerts, ${unreadAlertCount} unread`}
          >
            <BellAlertIcon className="w-5 h-5" />
            <span className="absolute -right-1 -top-1 min-w-4 h-4 px-1 rounded-full bg-red-500 text-white text-[9px] leading-4 text-center font-bold">
              {unreadAlertCount > 99 ? "99+" : unreadAlertCount}
            </span>
          </button>
          {alerts.length > 0 && (
            <div className="absolute right-0 top-10 z-20 hidden group-hover:block w-72 bg-white border border-slate-200 rounded-xl shadow-lg p-3">
              <div className="flex items-center justify-between mb-2"><p className="text-xs font-semibold text-navy-800">Alert Message</p><span className="text-[10px] text-navy-400">{unreadAlertCount} unread</span></div>
              <div className="space-y-2">{alerts.slice(0, 3).map((alert) => <button key={alert.id} type="button" onClick={() => { markAlertRead(alert.id); setPage("alerts"); }} className="w-full text-left flex items-start gap-2 p-1.5 rounded hover:bg-slate-50"><BellAlertIcon className="w-4 h-4 shrink-0 text-accent-blue mt-0.5" /><span className="min-w-0"><span className="block text-[11px] font-semibold text-navy-800 truncate">{alert.title}</span><span className="block text-[10px] text-navy-400 truncate">{alert.message}</span></span></button>)}</div>
              <button type="button" onClick={() => setPage("alerts")} className="w-full flex items-center justify-end gap-1 mt-2 pt-2 border-t border-slate-100 text-[11px] font-semibold text-accent-blue">View All Alerts <ChevronRightIcon className="w-3 h-3" /></button>
            </div>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {backendStatus === "loading" ? (
            <SignalIcon className="w-3.5 h-3.5 text-navy-300 animate-pulse" />
          ) : backendStatus === "ok" ? (
            <WifiIcon className="w-3.5 h-3.5 text-green-500" />
          ) : (
            <WifiIcon className="w-3.5 h-3.5 text-red-500" />
          )}
          <span className={`text-xs font-medium ${backendStatus === "ok" ? "text-green-600" : backendStatus === "error" ? "text-red-500" : "text-navy-400"}`}>
            {backendStatus === "ok" ? "Connected" : backendStatus === "error" ? "Offline" : "Checking"}
          </span>
        </div>

      </div>
    </header>
  );
}