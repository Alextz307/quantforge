import { useMemo, useState } from "react";
import { useMe } from "@/api/auth";
import {
  useStudiesPage,
  usePrefetchStudy,
  type StudiesPage as StudiesPageDto,
  type StudySummary,
} from "@/api/studies";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { Pagination } from "@/components/Pagination";
import { QueryRenderer } from "@/components/QueryRenderer";
import { usePaginatedSearch } from "@/hooks/usePaginatedSearch";
import { ALL_OPTION, readValidSince, withActiveOption } from "@/lib/filters";
import { formatDateTime, formatPercent } from "@/lib/format";
import { studyDetailPath } from "@/lib/routes";

interface StudiesUrlState {
  spec: string;
  since: string;
}

function readState(params: URLSearchParams): StudiesUrlState {
  return {
    spec: params.get("spec") ?? ALL_OPTION,
    since: readValidSince(params.get("since")),
  };
}

export function StudiesPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const { searchParams, limit, offset, setOffset, setParams } = usePaginatedSearch();
  const urlState = useMemo(() => readState(searchParams), [searchParams]);

  const query = useStudiesPage(
    {
      limit,
      offset,
      ...(urlState.spec !== ALL_OPTION ? { spec: urlState.spec } : {}),
      ...(urlState.since ? { since: new Date(urlState.since).toISOString() } : {}),
    },
    { allUsers: isAdmin && allUsers },
  );

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
          {(page) => (
            <StudiesBody
              page={page}
              spec={urlState.spec}
              since={urlState.since}
              onSpec={(v) => {
                setParams({ spec: v === ALL_OPTION ? "" : v });
              }}
              onSince={(v) => {
                setParams({ since: v });
              }}
              limit={limit}
              onOffset={setOffset}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  page: StudiesPageDto;
  spec: string;
  since: string;
  onSpec: (v: string) => void;
  onSince: (v: string) => void;
  limit: number;
  onOffset: (offset: number) => void;
}

function StudiesBody({ page, spec, since, onSpec, onSince, limit, onOffset }: BodyProps) {
  const specOptions = useMemo(() => withActiveOption(page.specs, spec), [page.specs, spec]);
  const prefetchStudy = usePrefetchStudy();

  return (
    <div className="flex flex-col gap-4">
      <FilterableTablePage<StudySummary>
        rows={page.items}
        filterControls={
          <>
            <FilterSelect
              id="filter-spec"
              label="Spec"
              value={spec}
              onChange={onSpec}
              allLabel="All specs"
              options={specOptions}
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
      <Pagination total={page.total} limit={limit} offset={page.offset} onOffset={onOffset} />
    </div>
  );
}
