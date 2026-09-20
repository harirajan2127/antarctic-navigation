import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type {
  AlertRecord,
  JourneyMode,
  JourneyResponse,
  JourneyStatus,
  PageId,
  RoutesOptimizeResponse,
} from "../types";
export const ALERT_CONFIG = {
  routeDistanceChangeKm: 50,
  routePathDeviationKm: 25,
};

interface Toast {
  id: number;
  message: string;
  type: "info" | "success" | "error";
}

interface AppCtx {
  page: PageId;
  setPage: (p: PageId) => void;
  alerts: AlertRecord[];
  unreadAlertCount: number;
  addAlert: (alert: Omit<AlertRecord, "id" | "timestamp" | "read"> & { id?: string }) => void;
    replaceRouteAlerts: (alerts: AlertRecord[]) => void;
  markAlertRead: (id: string, read?: boolean) => void;
  markAllAlertsRead: () => void;
  deleteAlert: (id: string) => void;
  clearAlerts: () => void;
  sidebarOpen: boolean;
  setSidebarOpen: (v: boolean) => void;
  toggleSidebar: () => void;
  toasts: Toast[];
  toast: (message: string, type?: Toast["type"]) => void;
  removeToast: (id: number) => void;
  activeRoute: RoutesOptimizeResponse | null;
  setActiveRoute: (route: RoutesOptimizeResponse | null) => void;
  navigation: NavigationSnapshot;
  setNavigation: (next: Partial<NavigationSnapshot>) => void;
}

export interface NavigationSnapshot {
  journey: JourneyResponse | null;
  status: JourneyStatus | null;
  planning: RoutesOptimizeResponse | null;
  portId: string;
  centerId: string;
  vesselId: string;
  mode: JourneyMode;
  liveLat: string;
  liveLon: string;
  selectedRoute: import("../types").RouteResult | null;
  alertFocus: [number, number] | null;
}

const Ctx = createContext<AppCtx>(null!);

let toastId = 0;

export function AppProvider({ children }: { children: ReactNode }) {
  const pageFromPath = (path: string): PageId => {
    switch (path) {
      case "/navigation":
        return "planner";
      case "/sea-ice":
        return "sea-ice";
      case "/icebergs":
        return "icebergs";
      case "/alert-message":
      case "/alerts":
        return "alerts";
      case "/assistant":
        return "assistant";
      default:
        return "home";
    }
  };

  const [page, setPageState] = useState<PageId>(() => pageFromPath(window.location.pathname));
  const [alerts, setAlerts] = useState<AlertRecord[]>(() => {
    try {
      const saved = window.localStorage.getItem("antarctic-dss-alerts");
      return saved ? (JSON.parse(saved) as AlertRecord[]) : [];
    } catch {
      return [];
    }
  });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const notifiedAlertIds = useRef(new Set<string>());
  const [activeRoute, setActiveRouteState] = useState<RoutesOptimizeResponse | null>(null);
  const [navigation, setNavigationState] = useState<NavigationSnapshot>({
    journey: null,
    status: null,
    planning: null,
    portId: "",
    centerId: "",
    vesselId: "polar_explorer",
    mode: "outbound",
    liveLat: "",
    liveLon: "",
    selectedRoute: null,
    alertFocus: null,
  });
  const setPage = useCallback((nextPage: PageId) => {
    setPageState(nextPage);
    const paths: Record<PageId, string> = {
      home: "/",
      planner: "/navigation",
      "sea-ice": "/sea-ice",
      icebergs: "/icebergs",
      alerts: "/alert-message",
      assistant: "/assistant",
    };
    window.history.pushState({}, "", paths[nextPage]);
  }, []);

  useEffect(() => {
    const onPopState = () => setPageState(pageFromPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);
  const setNavigation = useCallback(
    (next: Partial<NavigationSnapshot>) => setNavigationState((current) => ({ ...current, ...next })),
    [],
  );

  const toast = useCallback(
    (message: string, type: Toast["type"] = "info") => {
      const id = ++toastId;
      setToasts((t) => [...t, { id, message, type }]);
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
    },
    [],
  );

  const replaceRouteAlerts = useCallback((nextAlerts: AlertRecord[]) => {
    const newAlerts = nextAlerts.filter((alert) => !notifiedAlertIds.current.has(alert.id));
    nextAlerts.forEach((alert) => notifiedAlertIds.current.add(alert.id));
    setAlerts((current) => [
      ...nextAlerts,
      ...current.filter((alert) => !["route", "sea-ice", "iceberg"].includes(alert.type)),
    ]);
    const priorityAlert = newAlerts.find((alert) => ["critical", "high", "medium"].includes(alert.severity));
    if (priorityAlert) toast(priorityAlert.message, priorityAlert.severity === "critical" || priorityAlert.severity === "high" ? "error" : "info");
  }, [toast]);

  const addAlert = useCallback(
    (input: Omit<AlertRecord, "id" | "timestamp" | "read"> & { id?: string }) => {
      const id = input.id ?? `${input.type}-${input.routeId ?? "event"}-${input.title}`;
      if (notifiedAlertIds.current.has(id)) return;
      notifiedAlertIds.current.add(id);
      setAlerts((current) => current.some((alert) => alert.id === id)
        ? current
        : [{ ...input, id, timestamp: new Date().toISOString(), read: false }, ...current]);
      toast(input.message, input.severity === "critical" || input.severity === "high" ? "error" : "info");
    },
    [toast],
  );

  const setActiveRoute = useCallback((route: RoutesOptimizeResponse | null) => {
    setActiveRouteState(route);
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("antarctic-dss-alerts", JSON.stringify(alerts));
    } catch {
      // Alerts remain available in memory when storage is unavailable.
    }
  }, [alerts]);

  const markAlertRead = useCallback((id: string, read = true) => {
    setAlerts((current) => current.map((alert) => alert.id === id ? { ...alert, read } : alert));
  }, []);
  const markAllAlertsRead = useCallback(() => setAlerts((current) => current.map((alert) => ({ ...alert, read: true }))), []);
  const deleteAlert = useCallback((id: string) => {
    setAlerts((current) => current.filter((alert) => alert.id !== id));
  }, []);
  const clearAlerts = useCallback(() => {
    notifiedAlertIds.current.clear();
    setAlerts([]);
  }, []);

  const toggleSidebar = useCallback(() => setSidebarOpen((o) => !o), []);

  const removeToast = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const value = useMemo<AppCtx>(
    () => ({
      page,
      setPage,
      alerts,
      unreadAlertCount: alerts.filter((alert) => !alert.read).length,
      addAlert,
        replaceRouteAlerts,
      markAlertRead,
      markAllAlertsRead,
      deleteAlert,
      clearAlerts,
      sidebarOpen,
      setSidebarOpen,
      toggleSidebar,
      toasts,
      toast,
      removeToast,
      activeRoute,
      setActiveRoute,
      navigation,
      setNavigation,
    }),
    [page, alerts, addAlert, replaceRouteAlerts, markAlertRead, markAllAlertsRead, deleteAlert, clearAlerts, sidebarOpen, toasts, toast, removeToast, toggleSidebar, activeRoute, navigation, setNavigation, setActiveRoute],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  return useContext(Ctx);
}