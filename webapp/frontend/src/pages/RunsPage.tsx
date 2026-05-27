import { useMemo, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { useMe } from "@/api/auth";
import {
  usePrefetchRun,
  useRunsPage,
  type RunSortBy,
  type RunsPage,
  type SortOrder,
} from "@/api/runs";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { QueryRenderer } from "@/components/QueryRenderer";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { formatDateTime, formatMetric } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";

const DEFAULT_LIMIT = 50;
// Free-text inputs commit to the URL instantly (snappy + shareable) but the
// query/fetch waits for the input to settle so the server isn't hit per
// keystroke and the React Query cache cannot accumulate unbounded keys.
const FILTER_DEBOUNCE_MS = 300;
const SORT_BY_VALUES: ReadonlySet<RunSortBy> = new Set([
  "created_at",
  "sharpe_mean",
  "calmar_mean",
]);
const ORDER_VALUES: ReadonlySet<SortOrder> = new Set(["asc", "desc"]);

interface RunsPageState {
  limit: number;
  offset: number;
  sortBy: RunSortBy;
  order: SortOrder;
  strategy: string;
  ticker: string;
  since: string;
}

function readState(params: URLSearchParams): RunsPageState {
  const limit = Number.parseInt(params.get("limit") ?? "", 10);
  const offset = Number.parseInt(params.get("offset") ?? "", 10);
  const sortBy = params.get("sort_by");
  const order = params.get("order");
  return {
    limit: Number.isFinite(limit) && limit > 0 ? limit : DEFAULT_LIMIT,
    offset: Number.isFinite(offset) && offset >= 0 ? offset : 0,
    sortBy:
      sortBy && SORT_BY_VALUES.has(sortBy as RunSortBy) ? (sortBy as RunSortBy) : "created_at",
    order: order && ORDER_VALUES.has(order as SortOrder) ? (order as SortOrder) : "desc",
    strategy: params.get("strategy") ?? "",
    ticker: params.get("ticker") ?? "",
    since: params.get("since") ?? "",
  };
}

function setParam(params: URLSearchParams, key: string, value: string): URLSearchParams {
  const next = new URLSearchParams(params);
  if (value === "") next.delete(key);
  else next.set(key, value);
  return next;
}

export function RunsPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const state = useMemo(() => readState(searchParams), [searchParams]);
  const debouncedStrategy = useDebouncedValue(state.strategy, FILTER_DEBOUNCE_MS);
  const debouncedTicker = useDebouncedValue(state.ticker, FILTER_DEBOUNCE_MS);
  const debouncedSince = useDebouncedValue(state.since, FILTER_DEBOUNCE_MS);

  const query = useRunsPage(
    {
      limit: state.limit,
      offset: state.offset,
      sortBy: state.sortBy,
      order: state.order,
      ...(debouncedStrategy ? { strategy: debouncedStrategy } : {}),
      ...(debouncedTicker ? { ticker: debouncedTicker } : {}),
      ...(debouncedSince ? { since: new Date(debouncedSince).toISOString() } : {}),
    },
    { allUsers: isAdmin && allUsers },
  );

  const updateParam = (key: string, value: string) => {
    setSearchParams(setParam(searchParams, key, value));
  };

  const updateFilter = (key: "strategy" | "ticker" | "since", value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value === "") next.delete(key);
    else next.set(key, value);
    next.delete("offset");
    setSearchParams(next);
  };

  const toggleSort = (col: RunSortBy) => {
    const next = new URLSearchParams(searchParams);
    const sameCol = state.sortBy === col;
    next.set("sort_by", col);
    next.set("order", sameCol && state.order === "desc" ? "asc" : "desc");
    next.delete("offset");
    setSearchParams(next);
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
            <RunsBody
              page={page}
              state={state}
              onToggleSort={toggleSort}
              onPrev={() => {
                updateParam("offset", String(Math.max(0, state.offset - state.limit)));
              }}
              onNext={() => {
                updateParam("offset", String(state.offset + state.limit));
              }}
            />
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
  onPrev: () => void;
  onNext: () => void;
}

function RunsBody({ page, state, onToggleSort, onPrev, onNext }: RunsBodyProps) {
  const prefetchRun = usePrefetchRun();
  const location = useLocation();
  const fromUrl = location.pathname + location.search;
  const { items, total, offset } = page;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + items.length);
  const hasPrev = offset > 0;
  const hasNext = offset + items.length < total;

  if (items.length === 0) {
    return <p className="text-sm text-muted-foreground">No runs match the current filters.</p>;
  }

  return (
    <>
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
              <SortableHeader
                label="Calmar"
                col="calmar_mean"
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
                <td className="py-2 pr-4 text-right font-mono">{formatMetric(r.sharpe_mean, 3)}</td>
                <td className="py-2 pr-4 text-right font-mono">{formatMetric(r.calmar_mean, 3)}</td>
                <td className="py-2 pr-0">
                  <LaunchedByCell username={r.launched_by_username} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          Showing {start}–{end} of {total}
        </span>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" disabled={!hasPrev} onClick={onPrev}>
            Previous
          </Button>
          <Button variant="outline" size="sm" disabled={!hasNext} onClick={onNext}>
            Next
          </Button>
        </div>
      </div>
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
  const indicator = active ? (state.order === "desc" ? " ↓" : " ↑") : "";
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
