import type { ReactNode } from "react";

export interface JourneyMetric {
  label: string;
  value: string;
  sub?: string | null;
  tone?: "normal" | "good" | "warn" | "bad";
}

const TONE_CLASS: Record<NonNullable<JourneyMetric["tone"]>, string> = {
  normal: "text-navy-900",
  good: "text-green-700",
  warn: "text-amber-700",
  bad: "text-red-600",
};

export function MetricBar({ metrics }: { metrics: JourneyMetric[] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-3">
      {metrics.map((m) => (
        <div
          key={m.label}
          className="bg-white rounded-xl border border-slate-200 shadow-card px-4 py-3"
        >
          <p className="text-[10px] font-semibold uppercase tracking-wider text-navy-400">
            {m.label}
          </p>
          <p className={`mt-1 text-sm font-bold ${TONE_CLASS[m.tone ?? "normal"]}`}>
            {m.value}
          </p>
          {m.sub && <p className="mt-0.5 text-[10px] text-navy-400">{m.sub}</p>}
        </div>
      ))}
    </div>
  );
}

export function MetricLabel({ children }: { children: ReactNode }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wider text-navy-400">
      {children}
    </span>
  );
}