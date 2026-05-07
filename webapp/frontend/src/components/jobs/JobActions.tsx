import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import type { JobRow } from "@/api/jobs";
import { useCancelJob, jobLogDownloadUrl } from "@/api/jobs";
import { runDetailPath } from "@/lib/routes";

interface JobActionsProps {
  job: JobRow;
}

export function JobActions({ job }: JobActionsProps) {
  const cancel = useCancelJob(job.id);
  const isRunning = job.status === "running";
  const completedRunId = job.status === "completed" ? job.experiment_id : null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        variant="destructive"
        size="sm"
        disabled={!isRunning || cancel.isPending}
        onClick={() => {
          if (!window.confirm(`Cancel job ${job.id}?`)) return;
          cancel.mutate();
        }}
      >
        {cancel.isPending ? "Cancelling…" : "Cancel"}
      </Button>
      <a
        href={jobLogDownloadUrl(job.id)}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-blue-600 hover:underline"
      >
        Download log
      </a>
      {completedRunId && (
        <Link
          to={runDetailPath(completedRunId)}
          className="text-sm text-blue-600 hover:underline"
          data-testid="job-view-run-link"
        >
          View run →
        </Link>
      )}
      {cancel.isError && <span className="text-sm text-rose-600">{cancel.error.message}</span>}
    </div>
  );
}
