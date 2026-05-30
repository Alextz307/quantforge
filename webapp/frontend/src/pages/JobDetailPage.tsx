import { useParams } from "react-router-dom";
import { BackLink } from "@/components/BackLink";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { JobActions } from "@/components/jobs/JobActions";
import { JobStatusPill } from "@/components/jobs/JobStatusPill";
import { LogPane } from "@/components/jobs/LogPane";
import { QueryRenderer } from "@/components/QueryRenderer";
import { useJob, type JobRow } from "@/api/jobs";
import { useJobStream } from "@/hooks/useJobStream";
import { formatDateTime } from "@/lib/format";
import { ROUTES } from "@/lib/routes";

export function JobDetailPage() {
  const { jobId = "" } = useParams<{ jobId: string }>();
  const query = useJob(jobId);

  return (
    <div className="space-y-4">
      <BackLink to={ROUTES.jobs}>All jobs</BackLink>
      <QueryRenderer query={query} errorTitle="Failed to load job">
        {(job) => <JobDetailBody job={job} />}
      </QueryRenderer>
    </div>
  );
}

function JobDetailBody({ job }: { job: JobRow }) {
  const stream = useJobStream(job.id, job.status);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <CardTitle className="font-mono text-base">{job.id}</CardTitle>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <JobStatusPill status={job.status} />
            <span>kind: {job.kind}</span>
            {job.pid !== null && <span>pid: {job.pid}</span>}
            {job.exit_code !== null && <span>exit: {job.exit_code}</span>}
            <span>started: {job.started_at ? formatDateTime(job.started_at) : "-"}</span>
            <span>finished: {job.finished_at ? formatDateTime(job.finished_at) : "-"}</span>
          </div>
        </div>
        <JobActions job={job} />
      </CardHeader>
      <CardContent>
        <LogPane lines={stream.logs} connection={stream.connection} />
      </CardContent>
    </Card>
  );
}
