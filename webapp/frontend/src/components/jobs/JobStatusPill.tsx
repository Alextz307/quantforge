import { cn } from "@/lib/cn";
import type { JobStatus } from "@/api/jobs";

interface JobStatusPillProps {
  status: JobStatus;
  className?: string | undefined;
}

const STATUS_STYLE: Record<JobStatus, string> = {
  queued: "bg-slate-100 text-slate-700 ring-slate-200",
  running: "bg-amber-100 text-amber-800 ring-amber-200",
  completed: "bg-emerald-100 text-emerald-800 ring-emerald-200",
  failed: "bg-rose-100 text-rose-800 ring-rose-200",
  cancelled: "bg-slate-100 text-slate-600 ring-slate-200",
};

export function JobStatusPill({ status, className }: JobStatusPillProps) {
  return (
    <span
      data-testid="job-status-pill"
      data-status={status}
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset",
        STATUS_STYLE[status],
        className,
      )}
    >
      {status}
    </span>
  );
}
