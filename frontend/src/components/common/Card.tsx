import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 shadow-card ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between px-5 pt-4 pb-2">
      <div>
        <h3 className="text-sm font-semibold text-navy-900">{title}</h3>
        {subtitle && (
          <p className="text-xs text-navy-400 mt-0.5">{subtitle}</p>
        )}
      </div>
      {action}
    </div>
  );
}

export function CardBody({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`px-5 pb-4 ${className}`}>{children}</div>;
}