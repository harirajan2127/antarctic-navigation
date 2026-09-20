import {
  ArrowDownIcon,
  ArrowRightIcon,
  BellAlertIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ChevronRightIcon,
  CircleStackIcon,
  CpuChipIcon,
  FlagIcon,
  GlobeAmericasIcon,
  MapIcon,
  SignalIcon,
  ShieldCheckIcon,
  SparklesIcon,
  TruckIcon,
} from "@heroicons/react/24/outline";
import { Antarctic3DGlobe, type GridData } from "../components/maps/Antarctic3DGlobe";
import { useApp } from "../context/AppContext";
import type { IcebergInfo } from "../types";

const HOME_GRID: GridData = (() => {
  const lat = [-75, -72, -69, -66, -63, -60];
  const lon = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180];
  const concentration = lat.map((latitude) =>
    lon.map((longitude) => {
      const southernBand = Math.max(0, (-latitude - 58) / 20);
      const sectorVariation = (Math.sin((longitude + 20) / 26) + 1) * 0.06;
      return Math.min(0.92, 0.22 + southernBand * 0.48 + sectorVariation);
    }),
  );
  return { lat, lon, concentration };
})();

const HOME_ICEBERGS: IcebergInfo[] = [
  { iceberg_id: "A23A", latitude: -62.02, longitude: -52.23, length_km: 74, width_km: 59, last_observed: "2023-12-21T00:00:00Z", source: "processed iceberg observations", demo: false },
  { iceberg_id: "B39", latitude: -70.74, longitude: -56.37, length_km: 15, width_km: 7, last_observed: "2023-12-21T00:00:00Z", source: "processed iceberg observations", demo: false },
  { iceberg_id: "C15", latitude: -65.93, longitude: 143.18, length_km: 26, width_km: 19, last_observed: "2023-12-21T00:00:00Z", source: "processed iceberg observations", demo: false },
  { iceberg_id: "B46", latitude: -63.73, longitude: -140.88, length_km: 33, width_km: 9, last_observed: "2023-12-21T00:00:00Z", source: "processed iceberg observations", demo: false },
];

const capabilities = [
  [SignalIcon, "Sea-Ice Forecasting", "Monitor and forecast Antarctic sea-ice conditions to identify safer navigation corridors."],
  [FlagIcon, "Iceberg Tracking", "Track iceberg locations, movement and predicted trajectories using environmental data."],
  [MapIcon, "Intelligent Route Planning", "Generate routes that consider sea ice, icebergs, ocean conditions and vessel constraints."],
  [ShieldCheckIcon, "Risk Assessment", "Evaluate route conditions and identify potential navigation hazards."],
  [BellAlertIcon, "Real-Time Alerts", "Receive route-specific alerts for nearby sea ice, icebergs and changing conditions."],
  [TruckIcon, "Vessel Intelligence", "Analyze vessel capabilities, speed, ice class and operational constraints for route planning."],
] as const;

const workflow = [
  ["01", "Data Collection", "Satellite, oceanographic, meteorological and iceberg datasets."],
  ["02", "Data Processing", "Clean, transform and integrate Antarctic environmental data."],
  ["03", "AI/ML Analysis", "Predict sea-ice conditions and iceberg movement."],
  ["04", "Route Optimization", "Generate routes based on environmental hazards and vessel capabilities."],
  ["05", "Navigation Decision Support", "Provide maps, alerts, risk information and route recommendations."],
] as const;

const dataSources = [
  [GlobeAmericasIcon, "Satellite Data"],
  [SignalIcon, "Sea-Ice Data"],
  [FlagIcon, "Iceberg Data"],
  [CircleStackIcon, "Oceanographic Data"],
  [ChartBarIcon, "Meteorological Data"],
  [MapIcon, "Bathymetry Data"],
  [TruckIcon, "Vessel Data"],
] as const;

function SectionHeading({ eyebrow, title, text }: { eyebrow: string; title: string; text: string }) {
  return (
    <div className="max-w-2xl">
      <p className="text-[11px] font-bold uppercase tracking-[0.2em] text-sky-600">{eyebrow}</p>
      <h2 className="mt-2 text-2xl font-bold tracking-tight text-navy-950 sm:text-3xl">{title}</h2>
      <p className="mt-3 text-sm leading-6 text-navy-500">{text}</p>
    </div>
  );
}

export function HomePage() {
  const { setPage } = useApp();

  return (
    <div className="-m-4 min-h-full overflow-hidden bg-[#f7fbfd] lg:-m-6">
      <section className="relative overflow-hidden border-b border-sky-100 bg-[linear-gradient(120deg,#f8fcff_0%,#e8f5fb_58%,#d7edf5_100%)] px-5 pb-10 pt-8 sm:px-8 lg:px-12 lg:pb-14 lg:pt-12">
        <div className="absolute -right-24 -top-32 h-80 w-80 rounded-full border-[28px] border-white/60" />
        <div className="relative mx-auto grid min-w-0 max-w-7xl items-center gap-8 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="max-w-2xl">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white/75 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-700">
              <SparklesIcon className="h-4 w-4" /> Antarctic operations intelligence
            </div>
            <h1 className="max-w-xl text-4xl font-bold tracking-tight text-navy-950 sm:text-5xl lg:text-6xl">
              Antarctic Navigation Decision Support System
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-8 text-navy-600">
              AI-Powered Sea-Ice, Iceberg and Route Intelligence for Polar Navigation
            </p>
            <p className="mt-4 max-w-xl text-sm leading-6 text-navy-500">
              AI-powered decision support for safer, smarter and more efficient navigation through Antarctic waters.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <button onClick={() => setPage("planner")} className="inline-flex items-center gap-2 rounded-lg bg-navy-900 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-navy-900/15 transition hover:bg-navy-800">
                Open Navigation Dashboard <ArrowRightIcon className="h-4 w-4" />
              </button>
              <a href="#capabilities" className="inline-flex items-center gap-2 rounded-lg border border-sky-200 bg-white/80 px-4 py-3 text-sm font-semibold text-navy-800 transition hover:border-sky-400 hover:bg-white">
                Explore System <ChevronRightIcon className="h-4 w-4" />
              </a>
            </div>
            <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2 text-xs text-navy-500">
              <span className="inline-flex items-center gap-2"><CheckCircleIcon className="h-4 w-4 text-emerald-600" /> Processed environmental data</span>
              <span className="inline-flex items-center gap-2"><CheckCircleIcon className="h-4 w-4 text-emerald-600" /> Vessel-aware planning</span>
            </div>
          </div>
          <div className="relative min-w-0 rounded-2xl border border-white/80 bg-navy-950/95 p-2 shadow-2xl shadow-sky-900/15">
            <div className="absolute left-5 top-5 z-10 rounded-lg border border-white/10 bg-navy-950/80 px-3 py-2 text-[11px] text-sky-100 backdrop-blur">
              <p className="font-semibold text-white">Southern Ocean situational view</p>
              <p className="mt-1 text-sky-200/70">Lightweight operational preview</p>
            </div>
            <Antarctic3DGlobe seaIce={HOME_GRID} icebergs={HOME_ICEBERGS} vesselPos={[-58, -30]} vesselLabel="Research vessel" height="min(58vw, 480px)" />
            <div className="absolute bottom-5 left-5 z-10 flex items-center gap-3 rounded-lg border border-white/10 bg-navy-950/80 px-3 py-2 text-[10px] text-sky-100 backdrop-blur">
              <span className="h-2 w-2 rounded-full bg-sky-400" /> Sea ice
              <span className="h-2 w-2 rounded-full bg-cyan-300" /> Icebergs
              <span className="h-2 w-2 rounded-full bg-emerald-400" /> Vessel
            </div>
          </div>
        </div>
      </section>

      <section id="capabilities" className="mx-auto max-w-7xl px-5 py-14 sm:px-8 lg:px-12">
        <SectionHeading eyebrow="One decision surface" title="Decision Support Capabilities" text="Bring environmental signals, vessel context and route intelligence together before a voyage enters Antarctic waters." />
        <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {capabilities.map(([Icon, title, description]) => (
            <article key={title} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-sky-200 hover:shadow-md">
              <Icon className="h-6 w-6 text-sky-600" />
              <h3 className="mt-5 text-sm font-bold text-navy-900">{title}</h3>
              <p className="mt-2 text-xs leading-5 text-navy-500">{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="border-y border-slate-200 bg-white px-5 py-14 sm:px-8 lg:px-12">
        <div className="mx-auto max-w-7xl">
          <SectionHeading eyebrow="From observation to action" title="How It Works" text="A clear operational chain turns Antarctic data into context for human navigation decisions." />
          <div className="mt-9 grid gap-3 md:grid-cols-5">
            {workflow.map(([number, title, description], index) => (
              <div key={number} className="relative">
                <div className="h-full rounded-xl border border-slate-200 bg-[#f8fbfc] p-4">
                  <span className="text-2xl font-bold text-sky-200">{number}</span>
                  <h3 className="mt-4 text-sm font-bold text-navy-900">{title}</h3>
                  <p className="mt-2 text-xs leading-5 text-navy-500">{description}</p>
                </div>
                {index < workflow.length - 1 && <ArrowRightIcon className="absolute -right-3 top-1/2 z-10 hidden h-5 w-5 -translate-y-1/2 text-sky-400 md:block" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-14 sm:px-8 lg:px-12">
        <SectionHeading eyebrow="Integrated Antarctic Data" title="The context behind every route" text="The platform surfaces the environmental and operational datasets already present in the project rather than hiding them behind a single score." />
        <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          {dataSources.map(([Icon, title]) => <div key={title} className="rounded-xl border border-slate-200 bg-white p-4 text-center shadow-sm"><Icon className="mx-auto h-6 w-6 text-sky-600" /><p className="mt-3 text-xs font-semibold leading-4 text-navy-700">{title}</p></div>)}
        </div>
      </section>

      <section className="border-y border-slate-200 bg-[#eef8fb] px-5 py-14 sm:px-8 lg:px-12">
        <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-[0.8fr_1.2fr] lg:items-center">
          <SectionHeading eyebrow="Navigation workflow" title="Keep the ship on water, keep the decision visible" text="Route planning connects the departure port to an Antarctic coastal access point. Any inland transfer to a research station remains a separate operational step." />
          <div className="rounded-2xl border border-sky-100 bg-white p-5 shadow-sm">
            {["Departure Port · Cape Town", "Southern Ocean", "Antarctic Coastal Access", "Route and risk assessment", "Research station transfer · Maitri Station"].map((step, index, items) => <div key={step} className="flex items-center gap-3"><div className={`flex min-h-11 flex-1 items-center rounded-lg px-4 text-sm font-semibold ${index === items.length - 1 ? "bg-navy-900 text-white" : "bg-sky-50 text-navy-800"}`}>{step}</div>{index < items.length - 1 && <ArrowDownIcon className="h-4 w-4 shrink-0 text-sky-500" />}</div>)}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-14 sm:px-8 lg:px-12">
        <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end"><SectionHeading eyebrow="Route intelligence" title="Choose the route profile that fits the mission" text="Every route remains a decision aid. Environmental conditions can change, and no route should be treated as guaranteed safe." /><button onClick={() => setPage("planner")} className="inline-flex shrink-0 items-center gap-2 self-start rounded-lg bg-sky-600 px-4 py-3 text-sm font-semibold text-white hover:bg-sky-700 sm:self-end">Open Route Planner <ArrowRightIcon className="h-4 w-4" /></button></div>
        <div className="mt-8 grid gap-3 md:grid-cols-3">
          {["LOW RISK", "MEDIUM RISK", "HIGH RISK"].map((title, index) => <div key={title} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><div className={`h-1.5 w-14 rounded-full ${index === 0 ? "bg-emerald-500" : index === 1 ? "bg-amber-500" : "bg-rose-500"}`} /><h3 className="mt-5 text-sm font-bold text-navy-900">{title}</h3><p className="mt-2 text-xs leading-5 text-navy-500">{index === 0 ? "Safer navigation corridor with lower environmental hazard exposure." : index === 1 ? "Balanced route considering distance and environmental conditions." : "Shorter or more challenging route with increased environmental hazards."}</p></div>)}
        </div>
      </section>

      <section className="border-y border-slate-200 bg-white px-5 py-14 sm:px-8 lg:px-12">
        <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-2">
          <div><SectionHeading eyebrow="Preview only" title="Intelligent Navigation Alerts" text="Alert patterns are generated from route conditions in the active dashboard. These examples illustrate how they are presented." /><div className="mt-6 space-y-3">{[["ICEBERG ALERT", "Nearby iceberg detected along the selected route.", "text-rose-600"], ["SEA-ICE ALERT", "High sea-ice concentration detected near the selected route.", "text-amber-600"], ["ROUTE ALERT", "Environmental conditions changed. Route assessment recommended.", "text-sky-600"]].map(([title, message, color]) => <div key={title} className="flex gap-3 rounded-xl border border-slate-200 bg-[#f8fbfc] p-4"><BellAlertIcon className={`h-5 w-5 shrink-0 ${color}`} /><div><p className={`text-[11px] font-bold tracking-wide ${color}`}>{title}</p><p className="mt-1 text-xs text-navy-600">{message}</p></div></div>)}</div></div>
          <div className="rounded-2xl bg-navy-950 p-7 text-white"><TruckIcon className="h-7 w-7 text-cyan-300" /><h2 className="mt-5 text-2xl font-bold">Vessel-Aware Navigation</h2><p className="mt-3 text-sm leading-6 text-sky-100/75">Route planning can consider vessel type, cruising speed, maximum speed, ice class, icebreaking capability, draft, fuel capacity, operational range and cargo capacity.</p><button onClick={() => setPage("planner")} className="mt-7 inline-flex items-center gap-2 rounded-lg bg-white px-4 py-3 text-sm font-semibold text-navy-900 hover:bg-sky-50">Explore Vessel Information <ArrowRightIcon className="h-4 w-4" /></button></div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-14 sm:px-8 lg:px-12"><div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center"><div><SectionHeading eyebrow="Purpose-built for the mission" title="Designed for Antarctic Research Operations" text="The system is designed to support research-vessel navigation by combining environmental intelligence, iceberg tracking, sea-ice analysis and route optimization in a single decision-support platform." /></div><div className="rounded-xl border border-sky-100 bg-sky-50 p-6"><CpuChipIcon className="h-6 w-6 text-sky-600" /><h3 className="mt-4 text-sm font-bold text-navy-900">Technology foundation</h3><div className="mt-4 flex flex-wrap gap-2">{["React", "TypeScript", "Vite", "Python", "FastAPI", "Machine Learning", "GIS / Geospatial", "SQLAlchemy"].map((item) => <span key={item} className="rounded-full border border-sky-200 bg-white px-3 py-1.5 text-[11px] font-semibold text-navy-700">{item}</span>)}</div></div></div></section>
    </div>
  );
}
