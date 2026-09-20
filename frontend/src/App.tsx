import { AppProvider, useApp } from "./context/AppContext";
import { Sidebar } from "./components/layout/Sidebar";
import { Header } from "./components/layout/Header";
import { ToastContainer } from "./components/common/Toast";
import { HomePage } from "./pages/Home";
import { SeaIceForecastPage } from "./pages/SeaIceForecast";
import { IcebergTrackingPage } from "./pages/IcebergTracking";
import { NavigationPlannerPage } from "./pages/NavigationPlanner";
import { AlertMessagePage } from "./pages/AlertMessage";
import { AssistantPage } from "./pages/Assistant";

function Page() {
  const { page } = useApp();
  switch (page) {
    case "home":
      return <HomePage />;
    case "sea-ice":
      return <SeaIceForecastPage />;
    case "icebergs":
      return <IcebergTrackingPage />;
    case "planner":
      return <NavigationPlannerPage />;
    case "alerts":
      return <AlertMessagePage />;
    case "assistant":
      return <AssistantPage />;
  }
}

function Shell() {
  return (
    <div className="min-h-screen bg-[#f7f8fa] text-navy-900 flex">
      <Sidebar />
      <div className="min-w-0 flex-1 lg:ml-64 flex flex-col min-h-screen">
        <Header />
        <main className="min-w-0 flex-1 p-4 lg:p-6 overflow-y-auto">
          <Page />
        </main>
        <footer className="px-6 py-3 text-[11px] text-navy-400 border-t border-slate-200 bg-white">
          Antarctic Navigation Decision Support System — Problem Statement 26059. Data sources and model status are
          always shown in-page. Predictions are estimates based on processed observations and should not be the sole
          basis for navigation safety decisions.
        </footer>
      </div>
      <ToastContainer />
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}