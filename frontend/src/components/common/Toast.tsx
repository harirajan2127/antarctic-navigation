import { useApp } from "../../context/AppContext";
import { XMarkIcon } from "@heroicons/react/24/outline";

export function ToastContainer() {
  const { toasts, removeToast } = useApp();

  return (
    <div className="fixed bottom-4 right-4 z-[2000] flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`pointer-events-auto flex items-start gap-2 px-4 py-3 rounded-lg shadow-lg border text-sm max-w-sm animate-slide-up
            ${
              t.type === "error"
                ? "bg-red-50 border-red-200 text-red-800"
                : t.type === "success"
                  ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                  : "bg-white border-slate-200 text-navy-800"
            }`}
        >
          <span className="flex-1 text-xs leading-relaxed">{t.message}</span>
          <button
            onClick={() => removeToast(t.id)}
            className="shrink-0 mt-0.5 opacity-50 hover:opacity-100"
          >
            <XMarkIcon className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}