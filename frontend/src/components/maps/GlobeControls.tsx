import { useRef, useState } from "react";

interface GlobeControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onToggleRotate: (on: boolean) => void;
}

export function GlobeControls({
  onZoomIn,
  onZoomOut,
  onReset,
  onToggleRotate,
}: GlobeControlsProps) {
  const [rotating, setRotating] = useState(false);
  const rotateRef = useRef(false);

  const toggle = () => {
    const next = !rotateRef.current;
    rotateRef.current = next;
    setRotating(next);
    onToggleRotate(next);
  };

  return (
    <div className="absolute top-2 right-2 z-[600] flex flex-col gap-1.5">
      <div className="flex flex-col overflow-hidden rounded-lg border border-slate-200 bg-white/95 shadow-card">
        <button
          onClick={onZoomIn}
          title="Zoom in"
          aria-label="Zoom in"
          className="flex h-7 w-7 items-center justify-center text-sm font-bold text-navy-700 hover:bg-slate-100"
        >
          +
        </button>
        <button
          onClick={onZoomOut}
          title="Zoom out"
          aria-label="Zoom out"
          className="flex h-7 w-7 items-center justify-center text-sm font-bold text-navy-700 hover:bg-slate-100 border-t border-slate-200"
        >
          −
        </button>
        <button
          onClick={onReset}
          title="Reset view"
          aria-label="Reset view"
          className="flex h-7 w-7 items-center justify-center text-[11px] font-semibold text-navy-700 hover:bg-slate-100 border-t border-slate-200"
        >
          ⌂
        </button>
        <button
          onClick={toggle}
          title={rotating ? "Stop rotation" : "Auto-rotate"}
          aria-label={rotating ? "Stop rotation" : "Auto-rotate"}
          className={`flex h-7 w-7 items-center justify-center text-[11px] font-semibold border-t border-slate-200 ${
            rotating ? "bg-blue-50 text-accent-blue" : "text-navy-700 hover:bg-slate-100"
          }`}
        >
          {rotating ? "◉" : "⟳"}
        </button>
      </div>
    </div>
  );
}