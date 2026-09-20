import { useApp } from "../../context/AppContext";
import {
  CogIcon,
  ExclamationTriangleIcon,
  FlagIcon,
  MapIcon,
  BellAlertIcon,
  ChatBubbleLeftEllipsisIcon,
  HomeIcon,
  Bars3Icon,
} from "@heroicons/react/24/outline";

const NAV_ITEMS: { id: import("../../types").PageId; label: string; icon: typeof HomeIcon }[] = [
  { id: "home", label: "Home", icon: HomeIcon },
  { id: "sea-ice", label: "Sea-Ice Forecast", icon: CogIcon },
  { id: "icebergs", label: "Iceberg Tracking", icon: FlagIcon },
  { id: "planner", label: "Navigation Dashboard", icon: MapIcon },
  { id: "alerts", label: "Alert Message", icon: BellAlertIcon },
  { id: "assistant", label: "AI Assistant", icon: ChatBubbleLeftEllipsisIcon },
];

export function Sidebar() {
  const { page, setPage, sidebarOpen, setSidebarOpen } = useApp();

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed top-0 left-0 z-40 h-full w-64 bg-white border-r border-slate-200 shadow-sidebar transition-transform duration-200
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          lg:translate-x-0`}
      >
        <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-100">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-accent-blue text-white">
            <CogIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-navy-900 leading-tight">
              Antarctic DSS
            </h1>
            <p className="text-[11px] text-navy-400 leading-tight">
              Navigation Support
            </p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="ml-auto lg:hidden text-navy-400 hover:text-navy-700"
          >
            <Bars3Icon className="w-5 h-5" />
          </button>
        </div>

        <nav className="px-3 py-3 space-y-0.5">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => {
                setPage(id);
                setSidebarOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                ${
                  page === id
                    ? "bg-accent-blue/10 text-accent-blue"
                    : "text-navy-600 hover:bg-slate-50 hover:text-navy-800"
                }`}
            >
              <Icon className="w-[18px] h-[18px] shrink-0" />
              {label}
            </button>
          ))}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 px-3 py-3 border-t border-slate-100">
          <p className="text-[11px] text-navy-400 text-center">
            Problem Statement 26059
          </p>
        </div>
      </aside>
    </>
  );
}