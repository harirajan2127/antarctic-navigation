export interface TimelineStep {
  key: string;
  label: string;
  value: string;
  detail?: string | null;
  state: "done" | "active" | "upcoming";
}

const STATE_DOT: Record<TimelineStep["state"], string> = {
  done: "bg-blue-600 border-blue-600 text-white",
  active: "bg-white border-blue-600 text-blue-600 animate-pulse",
  upcoming: "bg-white border-slate-300 text-slate-400",
};

export function JourneyTimeline({ steps }: { steps: TimelineStep[] }) {
  return (
    <ol className="flex flex-col lg:flex-row lg:items-stretch gap-0">
      {steps.map((s, i) => {
        const last = i === steps.length - 1;
        return (
          <li key={s.key} className="relative flex flex-1 gap-3 lg:gap-0">
            {/* connector */}
            {!last && (
              <span className="hidden lg:block absolute top-5 left-7 right-0 h-0.5 bg-slate-200" />
            )}
            <div className="relative z-10 flex flex-col items-center lg:flex-1">
              <span
                className={`w-10 h-10 rounded-full border-2 flex items-center justify-center text-xs font-bold ${STATE_DOT[s.state]}`}
              >
                {i + 1}
              </span>
              <div className="mt-2 px-1 text-center">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-navy-400">
                  {s.label}
                </p>
                <p className="mt-0.5 text-xs font-bold text-navy-900 leading-snug">
                  {s.value}
                </p>
                {s.detail && (
                  <p className="mt-0.5 text-[10px] text-navy-400 leading-snug">
                    {s.detail}
                  </p>
                )}
              </div>
            </div>
            {/* connector small screens */}
            {!last && <span className="lg:hidden absolute left-5 top-10 bottom-0 w-0.5 bg-slate-200" />}
          </li>
        );
      })}
    </ol>
  );
}