export function Legend({
  title,
  stops,
}: {
  title: string;
  stops: { label: string; color: string }[];
}) {
  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-lg border border-slate-200 px-3 py-2">
      <p className="text-[10px] font-semibold text-navy-600 uppercase tracking-wider mb-1.5">
        {title}
      </p>
      <div className="flex items-center gap-1">
        {stops.map((s, i) => (
          <div key={i} className="flex items-center gap-1">
            <div
              className="w-4 h-2.5 rounded-sm"
              style={{ backgroundColor: s.color }}
            />
            <span className="text-[10px] text-navy-500">{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}