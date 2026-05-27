import { cn } from "@/lib/cn";
import type { JobStatus } from "@/api/jobs";

interface JobStatusPillProps {
  status: JobStatus;
  className?: string | undefined;
}

const STATUS_STYLE: Record<JobStatus, string> = {
  queued:
    "bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-800/60 dark:text-slate-200 dark:ring-slate-700",
  running:
    "bg-amber-100 text-amber-800 ring-amber-200 dark:bg-amber-900/40 dark:text-amber-200 dark:ring-amber-800/60",
  completed:
    "bg-emerald-100 text-emerald-800 ring-emerald-200 dark:bg-emerald-900/40 dark:text-emerald-200 dark:ring-emerald-800/60",
  failed:
    "bg-rose-100 text-rose-800 ring-rose-200 dark:bg-rose-900/40 dark:text-rose-200 dark:ring-rose-800/60",
  cancelled:
    "bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800/60 dark:text-slate-300 dark:ring-slate-700",
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
