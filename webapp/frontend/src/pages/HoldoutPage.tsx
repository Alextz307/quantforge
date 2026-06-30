import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMe } from "@/api/auth";
import { useCreateDeployment } from "@/api/deployments";
import {
  useHoldoutEvalsPage,
  usePrefetchHoldoutEval,
  type HoldoutEvalsPage,
  type HoldoutEvalSummary,
  type HoldoutSortBy,
} from "@/api/holdout";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import {
  FilterableTablePage,
  type SortOrder,
  type SortState,
} from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { Pagination } from "@/components/Pagination";
import { QueryRenderer } from "@/components/QueryRenderer";
import { usePaginatedSearch } from "@/hooks/usePaginatedSearch";
import { ALL_OPTION, readValidSince, withActiveOption } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { deploymentDetailPath, holdoutDetailPath, ROUTES } from "@/lib/routes";
import { SOURCE_KINDS, sourceKindLabel, type SourceKind } from "@/lib/sourceKind";

type SourceKindFilter = SourceKind | typeof ALL_OPTION;

const DEFAULT_SORT: SortState<HoldoutSortBy> = { sortBy: "created_at", order: "desc" };
const SORT_KEYS: ReadonlySet<HoldoutSortBy> = new Set([
  "created_at",
  "holdout_start",
  "sharpe_ratio",
]);
const ORDER_VALUES: ReadonlySet<SortOrder> = new Set(["asc", "desc"]);

function isSourceKindFilter(value: string): value is SourceKindFilter {
  return value === ALL_OPTION || (SOURCE_KINDS as readonly string[]).includes(value);
}

interface HoldoutUrlState {
  sortBy: HoldoutSortBy;
  order: SortOrder;
  sourceKind: SourceKindFilter;
  since: string;
}

// Sort + filters live in the URL so they survive a round-trip into a detail and
// back; allUsers stays component-local, matching the runs page.
function readState(params: URLSearchParams): HoldoutUrlState {
  const sortBy = params.get("sort_by");
  const order = params.get("order");
  const sourceKind = params.get("source_kind");
  return {
    sortBy:
      sortBy && SORT_KEYS.has(sortBy as HoldoutSortBy)
        ? (sortBy as HoldoutSortBy)
        : DEFAULT_SORT.sortBy,
    order:
      order && ORDER_VALUES.has(order as SortOrder) ? (order as SortOrder) : DEFAULT_SORT.order,
    sourceKind: sourceKind && isSourceKindFilter(sourceKind) ? sourceKind : ALL_OPTION,
    since: readValidSince(params.get("since")),
  };
}

export function HoldoutPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const { searchParams, limit, offset, setOffset, setParams } = usePaginatedSearch();
  const urlState = useMemo(() => readState(searchParams), [searchParams]);
  const sortState = useMemo<SortState<HoldoutSortBy>>(
    () => ({ sortBy: urlState.sortBy, order: urlState.order }),
    [urlState.sortBy, urlState.order],
  );

  const query = useHoldoutEvalsPage(
    {
      limit,
      offset,
      sortBy: urlState.sortBy,
      order: urlState.order,
      ...(urlState.sourceKind !== ALL_OPTION ? { sourceKind: urlState.sourceKind } : {}),
      ...(urlState.since ? { since: new Date(urlState.since).toISOString() } : {}),
    },
    { allUsers: isAdmin && allUsers },
  );

  const onSortToggle = (col: HoldoutSortBy) => {
    setParams({
      sort_by: col,
      order: urlState.sortBy === col && urlState.order === "desc" ? "asc" : "desc",
    });
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Holdout evaluations</CardTitle>
        <Button asChild size="sm">
          <Link to={ROUTES.configureHoldout} data-testid="holdout-new-cta">
            New holdout eval
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="holdout evaluations"
          testId="holdout-all-users-toggle"
        />
        <QueryRenderer query={query} errorTitle="Failed to load holdout evaluations">
          {(page) => (
            <HoldoutBody
              page={page}
              sourceKind={urlState.sourceKind}
              since={urlState.since}
              onSourceKind={(v) => {
                setParams({ source_kind: v === ALL_OPTION ? "" : v });
              }}
              onSince={(v) => {
                setParams({ since: v });
              }}
              sortState={sortState}
              onSortToggle={onSortToggle}
              limit={limit}
              offset={offset}
              onOffset={setOffset}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  page: HoldoutEvalsPage;
  sourceKind: SourceKindFilter;
  since: string;
  onSourceKind: (v: SourceKindFilter) => void;
  onSince: (v: string) => void;
  sortState: SortState<HoldoutSortBy>;
  onSortToggle: (col: HoldoutSortBy) => void;
  limit: number;
  offset: number;
  onOffset: (offset: number) => void;
}

function HoldoutBody({
  page,
  sourceKind,
  since,
  onSourceKind,
  onSince,
  sortState,
  onSortToggle,
  limit,
  offset,
  onOffset,
}: BodyProps) {
  const sourceKindOptions = useMemo(
    () => withActiveOption(page.source_kinds, sourceKind),
    [page.source_kinds, sourceKind],
  );
  const prefetchHoldoutEval = usePrefetchHoldoutEval();
  const create = useCreateDeployment();
  const navigate = useNavigate();

  function deploy(source: HoldoutEvalSummary) {
    create.mutate(
      { source_kind: source.source_kind, source_id: source.source_id },
      {
        onSuccess: (d) => {
          navigate(deploymentDetailPath(d.id));
        },
      },
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {create.isError && (
        <Alert variant="destructive">
          <AlertDescription>{create.error.message}</AlertDescription>
        </Alert>
      )}
      <FilterableTablePage<HoldoutEvalSummary, Record<string, never>, HoldoutSortBy>
        rows={page.items}
        filters={{}}
        applyFilters={(rows) => rows}
        filterControls={
          <>
            <FilterSelect
              id="filter-source-kind"
              label="Source kind"
              value={sourceKind}
              onChange={(next) => {
                if (isSourceKindFilter(next)) onSourceKind(next);
              }}
              allLabel="All source kinds"
              options={sourceKindOptions}
              optionLabel={(v) => sourceKindLabel(v as SourceKind)}
            />
            <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
          </>
        }
        rowKey={(r) => r.name}
        rowName={(r) => r.name}
        rowHref={(r) => holdoutDetailPath(r.name)}
        rowOnHover={(r) => {
          prefetchHoldoutEval(r.name);
        }}
        tableTestId="holdout-table"
        emptyMessage="No holdout evaluations match the current filters."
        sortState={sortState}
        onSortToggle={onSortToggle}
        columns={[
          { header: "Store", cellClassName: "font-mono", render: (r) => r.store },
          {
            header: "Source",
            cellClassName: "font-mono text-xs",
            render: (r) => `${sourceKindLabel(r.source_kind)} | ${r.source_id}`,
          },
          {
            header: "Sharpe",
            align: "right",
            cellClassName: "font-mono",
            render: (r) => formatMetric(r.sharpe_ratio),
            sortKey: "sharpe_ratio",
          },
          {
            header: "Holdout start",
            cellClassName: "font-mono text-xs",
            render: (r) => formatDateTime(r.holdout_start),
            sortKey: "holdout_start",
          },
          {
            header: "Created",
            cellClassName: "font-mono text-xs",
            render: (r) => formatDateTime(r.created_at),
            sortKey: "created_at",
          },
          {
            header: "Launched by",
            render: (r) => <LaunchedByCell username={r.launched_by_username} />,
          },
          {
            header: "",
            align: "right",
            render: (r) => (
              <Button
                size="sm"
                variant="outline"
                disabled={create.isPending}
                onClick={() => {
                  deploy(r);
                }}
                data-testid={`deploy-holdout-${r.name}`}
              >
                Deploy
              </Button>
            ),
          },
        ]}
      />
      {(page.items.length > 0 || offset > 0) && (
        <Pagination
          total={page.total}
          limit={limit}
          offset={offset}
          count={page.items.length}
          onOffset={onOffset}
        />
      )}
    </div>
  );
}
