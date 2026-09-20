import type { ReactNode } from "react";

export function StatCard({
  label,
  value,
  sub,
  icon,
  accent = "blue",
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
  accent?: "blue" | "green" | "amber" | "red" | "cyan";
}) {
  const accents = {
    blue: "bg-ice-100 text-ice-700",
    green: "bg-emerald-50 text-emerald-600",
    amber: "bg-amber-50 text-amber-600",
    red: "bg-red-50 text-red-600",
    cyan: "bg-cyan-50 text-cyan-600",
  };
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-card p-4 flex items-start gap-3">
      {icon && (
        <div className={`p-2 rounded-lg ${accents[accent]}`}>{icon}</div>
      )}
      <div className="min-w-0">
        <p className="text-xs font-medium text-navy-400 truncate">{label}</p>
        <p className="text-lg font-bold text-navy-900 leading-tight mt-0.5">
          {value}
        </p>
        {sub && <p className="text-[11px] text-navy-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}