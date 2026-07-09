import { useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useMe } from "@/api/auth";
import {
  usePrefetchRun,
  useRunsPage,
  type RunSortBy,
  type RunsPage,
  type SortOrder,
} from "@/api/runs";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { Pagination } from "@/components/Pagination";
import { QueryRenderer } from "@/components/QueryRenderer";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { usePaginatedSearch } from "@/hooks/usePaginatedSearch";
import { readSortState, readValidSince, toggleSortParams } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";

// Free-text inputs commit to the URL instantly (snappy + shareable) but the
// query/fetch waits for the input to settle so the server isn't hit per
// keystroke and the React Query cache cannot accumulate unbounded keys.
const FILTER_DEBOUNCE_MS = 300;
const SORT_KEYS: ReadonlySet<RunSortBy> = new Set(["created_at", "sharpe_mean"]);
const DEFAULT_SORT: { sortBy: RunSortBy; order: "asc" | "desc" } = {
  sortBy: "created_at",
  order: "desc",
};

interface RunsFilters {
  sortBy: RunSortBy;
  order: SortOrder;
  strategy: string;
  ticker: string;
  since: string;
}

interface RunsPageState extends RunsFilters {
  limit: number;
  offset: number;
}

function readFilters(params: URLSearchParams): RunsFilters {
  return {
    ...readSortState(params, SORT_KEYS, DEFAULT_SORT),
    strategy: params.get("strategy") ?? "",
    ticker: params.get("ticker") ?? "",
    since: readValidSince(params.get("since")),
  };
}

export function RunsPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const { searchParams, limit, offset, setOffset, setParams } = usePaginatedSearch();
  const filters = useMemo(() => readFilters(searchParams), [searchParams]);
  const state: RunsPageState = useMemo(
    () => ({ limit, offset, ...filters }),
    [limit, offset, filters],
  );
  const debouncedStrategy = useDebouncedValue(filters.strategy, FILTER_DEBOUNCE_MS);
  const debouncedTicker = useDebouncedValue(filters.ticker, FILTER_DEBOUNCE_MS);
  const debouncedSince = useDebouncedValue(filters.since, FILTER_DEBOUNCE_MS);

  const query = useRunsPage(
    {
      limit,
      offset,
      sortBy: filters.sortBy,
      order: filters.order,
      ...(debouncedStrategy ? { strategy: debouncedStrategy } : {}),
      ...(debouncedTicker ? { ticker: debouncedTicker } : {}),
      ...(debouncedSince ? { since: new Date(debouncedSince).toISOString() } : {}),
    },
    { allUsers: isAdmin && allUsers },
  );

  const updateFilter = (key: "strategy" | "ticker" | "since", value: string) => {
    setParams({ [key]: value });
  };

  const toggleSort = (col: RunSortBy) => {
    setParams(toggleSortParams(filters, col));
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="runs"
          testId="runs-all-users-toggle"
        />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex flex-col gap-1">
            <Label htmlFor="filter-strategy">Strategy</Label>
            <Input
              id="filter-strategy"
              value={state.strategy}
              placeholder="e.g. VolatilityTargeting"
              onChange={(e) => {
                updateFilter("strategy", e.target.value);
              }}
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label htmlFor="filter-ticker">Ticker</Label>
            <Input
              id="filter-ticker"
              value={state.ticker}
              placeholder="e.g. SPY"
              onChange={(e) => {
                updateFilter("ticker", e.target.value);
              }}
            />
          </div>
          <FilterDate
            id="filter-since"
            label="Since"
            value={state.since}
            onChange={(v) => {
              updateFilter("since", v);
            }}
          />
        </div>

        <QueryRenderer query={query} errorTitle="Failed to load runs">
          {(page) => (
            <RunsBody page={page} state={state} onToggleSort={toggleSort} onOffset={setOffset} />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface RunsBodyProps {
  page: RunsPage;
  state: RunsPageState;
  onToggleSort: (col: RunSortBy) => void;
  onOffset: (offset: number) => void;
}

function RunsBody({ page, state, onToggleSort, onOffset }: RunsBodyProps) {
  const prefetchRun = usePrefetchRun();
  const location = useLocation();
  const fromUrl = location.pathname + location.search;
  const { items, total, offset } = page;

  return (
    <>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground">No runs match the current filters.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="runs-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4 font-mono">Strategy</th>
                <th className="py-2 pr-4 font-mono">Tickers</th>
                <th className="py-2 pr-4 font-mono">Interval</th>
                <SortableHeader
                  label="Created"
                  col="created_at"
                  state={state}
                  onToggle={onToggleSort}
                />
                <SortableHeader
                  label="Sharpe"
                  col="sharpe_mean"
                  state={state}
                  onToggle={onToggleSort}
                  align="right"
                />
                <th className="py-2 pr-0">Launched by</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr
                  key={r.experiment_id}
                  className="border-b last:border-0"
                  onMouseEnter={() => {
                    prefetchRun(r.experiment_id);
                  }}
                >
                  <td className="py-2 pr-4">
                    <Link
                      to={runDetailPath(r.experiment_id)}
                      state={{ from: fromUrl }}
                      className="text-primary hover:underline"
                    >
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.strategy}</td>
                  <td className="py-2 pr-4 font-mono">{r.tickers.join(", ")}</td>
                  <td className="py-2 pr-4 font-mono">{r.interval}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{formatDateTime(r.created_at)}</td>
                  <td className="py-2 pr-4 text-right font-mono">
                    {formatMetric(r.sharpe_mean, 3)}
                  </td>
                  <td className="py-2 pr-0">
                    <LaunchedByCell username={r.launched_by_username} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Pagination total={total} limit={state.limit} offset={offset} onOffset={onOffset} />
    </>
  );
}

interface SortableHeaderProps {
  label: string;
  col: RunSortBy;
  state: RunsPageState;
  onToggle: (col: RunSortBy) => void;
  align?: "left" | "right";
  isLast?: boolean;
}

function SortableHeader({
  label,
  col,
  state,
  onToggle,
  align = "left",
  isLast,
}: SortableHeaderProps) {
  const active = state.sortBy === col;
  const indicator = active ? (state.order === "desc" ? " v" : " ^") : "";
  const padRight = isLast ? "pr-0" : "pr-4";
  const alignCls = align === "right" ? "text-right" : "text-left";

  return (
    <th className={`py-2 ${padRight} ${alignCls}`}>
      <button
        type="button"
        className={`hover:text-foreground ${active ? "text-foreground" : ""}`}
        onClick={() => {
          onToggle(col);
        }}
      >
        {label}
        {indicator}
      </button>
    </th>
  );
}
