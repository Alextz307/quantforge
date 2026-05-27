import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMe } from "@/api/auth";
import { useComparisons, usePrefetchComparison, type ComparisonSummary } from "@/api/comparisons";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { comparisonDetailPath, ROUTES } from "@/lib/routes";

interface ComparisonsFilters {
  strategy: string;
  since: string;
}

function applyFilters(
  rows: readonly ComparisonSummary[],
  f: ComparisonsFilters,
): readonly ComparisonSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.strategy !== ALL_OPTION && !r.strategies.includes(f.strategy)) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function ComparisonsPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const query = useComparisons({ allUsers: isAdmin && allUsers });
  const [strategy, setStrategy] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

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
          {(rows) => (
            <ComparisonsBody
              rows={rows}
              strategy={strategy}
              since={since}
              onStrategy={setStrategy}
              onSince={setSince}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly ComparisonSummary[];
  strategy: string;
  since: string;
  onStrategy: (v: string) => void;
  onSince: (v: string) => void;
}

function ComparisonsBody({ rows, strategy, since, onStrategy, onSince }: BodyProps) {
  const strategies = useMemo(() => uniqSorted(rows.flatMap((r) => r.strategies)), [rows]);
  const filters = useMemo<ComparisonsFilters>(() => ({ strategy, since }), [strategy, since]);
  const prefetchComparison = usePrefetchComparison();

  return (
    <FilterableTablePage<ComparisonSummary, ComparisonsFilters>
      rows={rows}
      filters={filters}
      applyFilters={applyFilters}
      filterControls={
        <>
          <FilterSelect
            id="filter-strategy"
            label="Strategy"
            value={strategy}
            onChange={onStrategy}
            allLabel="All strategies"
            options={strategies}
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
  );
}
