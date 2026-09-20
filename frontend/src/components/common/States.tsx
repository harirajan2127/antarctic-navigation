export function Spinner({ size = "md", className = "" }: { size?: "sm" | "md" | "lg"; className?: string }) {
  const px = size === "sm" ? "w-4 h-4" : size === "lg" ? "w-10 h-10" : "w-6 h-6";
  return (
    <div className={`${px} border-2 border-slate-200 border-t-accent-blue rounded-full animate-spin ${className}`} />
  );
}

export function LoadingState({ message = "Loading..." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-navy-400">
      <Spinner size="lg" />
      <p className="text-sm mt-3">{message}</p>
    </div>
  );
}

export function ErrorState({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="w-10 h-10 rounded-full bg-red-50 flex items-center justify-center mb-3">
        <span className="text-red-500 text-lg">!</span>
      </div>
      <p className="text-sm text-navy-700 font-medium mb-1">Something went wrong</p>
      <p className="text-xs text-navy-400 max-w-xs">{message}</p>
      {retry && (
        <button
          onClick={retry}
          className="mt-3 px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-navy-700 rounded-lg transition-colors"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <p className="text-sm text-navy-500 font-medium">{title}</p>
      {description && (
        <p className="text-xs text-navy-400 max-w-xs mt-1">{description}</p>
      )}
    </div>
  );
}