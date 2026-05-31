import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMe } from "@/api/auth";
import { useJobs, type JobRow, type JobStatus } from "@/api/jobs";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { JobStatusPill } from "@/components/jobs/JobStatusPill";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { jobDetailPath, ROUTES } from "@/lib/routes";
import { JobArtifactLink } from "@/components/jobs/JobArtifactLink";

interface JobFilters {
  status: string;
}

const JOB_STATUSES: readonly JobStatus[] = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
];

function applyFilters(jobs: readonly JobRow[], f: JobFilters): readonly JobRow[] {
  if (f.status === ALL_OPTION) return jobs;
  return jobs.filter((j) => j.status === f.status);
}

export function JobsPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const jobsQuery = useJobs({ allUsers: isAdmin && allUsers });
  const [status, setStatus] = useState<string>(ALL_OPTION);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-4">
        <CardTitle>Jobs</CardTitle>
        <Link
          to={ROUTES.configure}
          className="text-sm text-primary hover:underline"
          data-testid="jobs-launch-link"
        >
          Launch new job {"->"}
        </Link>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <QueryRenderer query={jobsQuery} errorTitle="Failed to load jobs">
          {(rows) => (
            <JobsBody
              rows={rows}
              status={status}
              onStatus={setStatus}
              isAdmin={isAdmin}
              allUsers={allUsers}
              onAllUsers={setAllUsers}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface JobsBodyProps {
  rows: readonly JobRow[];
  status: string;
  onStatus: (v: string) => void;
  isAdmin: boolean;
  allUsers: boolean;
  onAllUsers: (v: boolean) => void;
}

function JobsBody({ rows, status, onStatus, isAdmin, allUsers, onAllUsers }: JobsBodyProps) {
  const filters = useMemo<JobFilters>(() => ({ status }), [status]);
  return (
    <>
      <AllUsersToggle
        isAdmin={isAdmin}
        checked={allUsers}
        onChange={onAllUsers}
        artifactLabel="jobs"
      />
      <FilterableTablePage<JobRow, JobFilters>
        rows={rows}
        filters={filters}
        applyFilters={applyFilters}
        filterGridClassName="md:grid-cols-2"
        filterControls={
          <FilterSelect
            id="filter-status"
            label="Status"
            value={status}
            onChange={onStatus}
            allLabel="All statuses"
            options={JOB_STATUSES}
          />
        }
        rowKey={(j) => j.id}
        rowName={(j) => j.id.slice(0, 8)}
        rowHref={(j) => jobDetailPath(j.id)}
        tableTestId="jobs-table"
        nameHeader="Job"
        emptyMessage="No jobs match the current filter."
        columns={[
          {
            header: "Status",
            render: (j) => <JobStatusPill status={j.status} />,
          },
          { header: "Kind", cellClassName: "font-mono", render: (j) => j.kind },
          {
            header: "Started",
            cellClassName: "font-mono text-xs",
            render: (j) => (j.started_at ? formatDateTime(j.started_at) : "-"),
          },
          {
            header: "Finished",
            cellClassName: "font-mono text-xs",
            render: (j) => (j.finished_at ? formatDateTime(j.finished_at) : "-"),
          },
          {
            header: "Artifact",
            render: (j) => <JobArtifactLink job={j} compact fallback="-" />,
          },
          {
            header: "Launched by",
            render: (j) => <LaunchedByCell username={j.launched_by_username} />,
          },
        ]}
      />
    </>
  );
}
