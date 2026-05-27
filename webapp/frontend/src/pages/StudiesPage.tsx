import { useMemo, useState } from "react";
import { useMe } from "@/api/auth";
import { useStudies, usePrefetchStudy, type StudySummary } from "@/api/studies";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatPercent } from "@/lib/format";
import { studyDetailPath } from "@/lib/routes";

interface StudiesFilters {
  spec: string;
  since: string;
}

function applyFilters(rows: readonly StudySummary[], f: StudiesFilters): readonly StudySummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.spec !== ALL_OPTION && r.spec_name !== f.spec) return false;
    if (sinceMs != null && new Date(r.started_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function StudiesPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const query = useStudies({ allUsers: isAdmin && allUsers });
  const [spec, setSpec] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Studies</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="studies"
          testId="studies-all-users-toggle"
        />
        <QueryRenderer query={query} errorTitle="Failed to load studies">
          {(rows) => (
            <StudiesBody
              rows={rows}
              spec={spec}
              since={since}
              onSpec={setSpec}
              onSince={setSince}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly StudySummary[];
  spec: string;
  since: string;
  onSpec: (v: string) => void;
  onSince: (v: string) => void;
}

function StudiesBody({ rows, spec, since, onSpec, onSince }: BodyProps) {
  const specs = useMemo(() => uniqSorted(rows.map((r) => r.spec_name)), [rows]);
  const filters = useMemo<StudiesFilters>(() => ({ spec, since }), [spec, since]);
  const prefetchStudy = usePrefetchStudy();

  return (
    <FilterableTablePage<StudySummary, StudiesFilters>
      rows={rows}
      filters={filters}
      applyFilters={applyFilters}
      filterControls={
        <>
          <FilterSelect
            id="filter-spec"
            label="Spec"
            value={spec}
            onChange={onSpec}
            allLabel="All specs"
            options={specs}
          />
          <FilterDate id="filter-since" label="Started since" value={since} onChange={onSince} />
        </>
      }
      rowKey={(r) => r.name}
      rowName={(r) => r.name}
      rowHref={(r) => studyDetailPath(r.name)}
      rowOnHover={(r) => {
        prefetchStudy(r.name);
      }}
      tableTestId="studies-table"
      emptyMessage="No studies match the current filters."
      columns={[
        { header: "Spec", cellClassName: "font-mono", render: (r) => r.spec_name },
        {
          header: "Legs",
          align: "right",
          cellClassName: "font-mono",
          render: (r) => `${String(r.completed_legs)} / ${String(r.total_legs)}`,
        },
        {
          header: "Completion",
          align: "right",
          cellClassName: "font-mono",
          render: (r) => formatPercent(r.completion_pct / 100),
        },
        {
          header: "Started",
          cellClassName: "font-mono text-xs",
          render: (r) => formatDateTime(r.started_at),
        },
        {
          header: "Launched by",
          render: (r) => <LaunchedByCell username={r.launched_by_username} />,
        },
      ]}
    />
  );
}
