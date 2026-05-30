import { Button } from "@/components/ui/button";
import type { JobRow } from "@/api/jobs";
import { useCancelJob, jobLogDownloadUrl } from "@/api/jobs";
import { JobArtifactLink } from "@/components/jobs/JobArtifactLink";

interface JobActionsProps {
  job: JobRow;
}

export function JobActions({ job }: JobActionsProps) {
  const cancel = useCancelJob(job.id);
  const isRunning = job.status === "running";

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
        {cancel.isPending ? "Cancelling..." : "Cancel"}
      </Button>
      <a
        href={jobLogDownloadUrl(job.id)}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-blue-600 hover:underline dark:text-blue-400"
      >
        Download log
      </a>
      <JobArtifactLink job={job} className="text-sm" />
      {cancel.isError && (
        <span className="text-sm text-rose-600 dark:text-rose-400">{cancel.error.message}</span>
      )}
    </div>
  );
}
