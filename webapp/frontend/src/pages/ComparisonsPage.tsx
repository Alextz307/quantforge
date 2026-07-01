import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMe } from "@/api/auth";
import {
  useComparisonsPage,
  usePrefetchComparison,
  type ComparisonsPage as ComparisonsPageDto,
  type ComparisonSummary,
} from "@/api/comparisons";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { Pagination } from "@/components/Pagination";
import { QueryRenderer } from "@/components/QueryRenderer";
import { usePaginatedSearch } from "@/hooks/usePaginatedSearch";
import { ALL_OPTION, readValidSince, withActiveOption } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { comparisonDetailPath, ROUTES } from "@/lib/routes";

interface ComparisonsUrlState {
  strategy: string;
  since: string;
}

function readState(params: URLSearchParams): ComparisonsUrlState {
  return {
    strategy: params.get("strategy") ?? ALL_OPTION,
    since: readValidSince(params.get("since")),
  };
}

export function ComparisonsPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const { searchParams, limit, offset, setOffset, setParams } = usePaginatedSearch();
  const urlState = useMemo(() => readState(searchParams), [searchParams]);

  const query = useComparisonsPage(
    {
      limit,
      offset,
      ...(urlState.strategy !== ALL_OPTION ? { strategy: urlState.strategy } : {}),
      ...(urlState.since ? { since: new Date(urlState.since).toISOString() } : {}),
    },
    { allUsers: isAdmin && allUsers },
  );

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Comparisons</CardTitle>
        <Button asChild size="sm">
          <Link to={ROUTES.configureCompare} data-testid="comparisons-new-cta">
            New comparison
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="comparisons"
          testId="comparisons-all-users-toggle"
        />
        <QueryRenderer query={query} errorTitle="Failed to load comparisons">
          {(page) => (
            <ComparisonsBody
              page={page}
              strategy={urlState.strategy}
              since={urlState.since}
              onStrategy={(v) => {
                setParams({ strategy: v === ALL_OPTION ? "" : v });
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
  page: ComparisonsPageDto;
  strategy: string;
  since: string;
  onStrategy: (v: string) => void;
  onSince: (v: string) => void;
  limit: number;
  onOffset: (offset: number) => void;
}

function ComparisonsBody({
  page,
  strategy,
  since,
  onStrategy,
  onSince,
  limit,
  onOffset,
}: BodyProps) {
  const strategyOptions = useMemo(
    () => withActiveOption(page.strategies, strategy),
    [page.strategies, strategy],
  );
  const prefetchComparison = usePrefetchComparison();

  return (
    <div className="flex flex-col gap-4">
      <FilterableTablePage<ComparisonSummary>
        rows={page.items}
        filterControls={
          <>
            <FilterSelect
              id="filter-strategy"
              label="Strategy"
              value={strategy}
              onChange={onStrategy}
              allLabel="All strategies"
              options={strategyOptions}
            />
            <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
          </>
        }
        rowKey={(r) => r.name}
        rowName={(r) => r.name}
        rowHref={(r) => comparisonDetailPath(r.name)}
        rowOnHover={(r) => {
          prefetchComparison(r.name);
        }}
        tableTestId="comparisons-table"
        emptyMessage="No comparisons match the current filters."
        columns={[
          { header: "Store", cellClassName: "font-mono", render: (r) => r.store },
          {
            header: "Strategies",
            cellClassName: "font-mono",
            render: (r) => r.strategies.join(", "),
          },
          {
            header: "Created",
            cellClassName: "font-mono text-xs",
            render: (r) => formatDateTime(r.created_at),
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
