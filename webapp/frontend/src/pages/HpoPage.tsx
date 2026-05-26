import { useCallback, useMemo, useState } from "react";
import { useHpoStudies, usePrefetchHpoStudy, type HpoSummary } from "@/api/hpo";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import {
  FilterableTablePage,
  type SortOrder,
  type SortState,
} from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { hpoDetailPath } from "@/lib/routes";

interface HpoFilters {
  store: string;
  since: string;
}

type HpoSortKey = "created_at" | "best_value";

const DEFAULT_SORT: SortState<HpoSortKey> = { sortBy: "created_at", order: "desc" };

function applyFilters(rows: readonly HpoSummary[], f: HpoFilters): readonly HpoSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.store !== ALL_OPTION && r.store !== f.store) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

function sortRows(
  rows: readonly HpoSummary[],
  state: SortState<HpoSortKey>,
): readonly HpoSummary[] {
  // Studies with no completed trials carry ``best_value=null``; under DESC
  // they sink to the bottom (and under ASC they float to the top, since a
  // null best_value is the "worst possible" reading).
  const dir = state.order === "desc" ? -1 : 1;
  const copied = [...rows];
  copied.sort((a, b) => {
    if (state.sortBy === "best_value") {
      const av = a.best_value;
      const bv = b.best_value;
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return (av - bv) * dir;
    }
    return (new Date(a.created_at).getTime() - new Date(b.created_at).getTime()) * dir;
  });
  return copied;
}

export function HpoPage() {
  const query = useHpoStudies();
  const [store, setStore] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");
  const [sortState, setSortState] = useState<SortState<HpoSortKey>>(DEFAULT_SORT);

  const onSortToggle = useCallback((col: HpoSortKey) => {
    setSortState((prev) => {
      // Re-clicking the active column flips order; switching columns starts in
      // DESC (the natural "best first" reading for both timestamps and Sharpe).
      const nextOrder: SortOrder = prev.sortBy === col && prev.order === "desc" ? "asc" : "desc";
      return { sortBy: col, order: nextOrder };
    });
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>HPO studies</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <QueryRenderer query={query} errorTitle="Failed to load HPO studies">
          {(rows) => (
            <HpoBody
              rows={rows}
              store={store}
              since={since}
              onStore={setStore}
              onSince={setSince}
              sortState={sortState}
              onSortToggle={onSortToggle}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly HpoSummary[];
  store: string;
  since: string;
  onStore: (v: string) => void;
  onSince: (v: string) => void;
  sortState: SortState<HpoSortKey>;
  onSortToggle: (col: HpoSortKey) => void;
}

function HpoBody({ rows, store, since, onStore, onSince, sortState, onSortToggle }: BodyProps) {
  const stores = useMemo(() => uniqSorted(rows.map((r) => r.store)), [rows]);
  const filters = useMemo<HpoFilters>(() => ({ store, since }), [store, since]);
  // Sort first, filter second: filtering doesn't change relative order so the
  // result is identical either way, but sort-then-filter keeps the
  // ``applyFilters`` signature pure (no sort plumbed through filter state).
  const sorted = useMemo(() => sortRows(rows, sortState), [rows, sortState]);
  const prefetchHpo = usePrefetchHpoStudy();

  return (
    <FilterableTablePage<HpoSummary, HpoFilters, HpoSortKey>
      rows={sorted}
      filters={filters}
      applyFilters={applyFilters}
      filterControls={
        <>
          <FilterSelect
            id="filter-store"
            label="Store"
            value={store}
            onChange={onStore}
            allLabel="All stores"
            options={stores}
          />
          <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
        </>
      }
      rowKey={(r) => r.name}
      rowName={(r) => r.name}
      rowHref={(r) => hpoDetailPath(r.name)}
      rowOnHover={(r) => {
        prefetchHpo(r.name);
      }}
      tableTestId="hpo-table"
      emptyMessage="No HPO studies match the current filters."
      sortState={sortState}
      onSortToggle={onSortToggle}
      columns={[
        { header: "Store", cellClassName: "font-mono", render: (r) => r.store },
        { header: "Direction", cellClassName: "font-mono", render: (r) => r.direction },
        {
          header: "Trials",
          align: "right",
          cellClassName: "font-mono",
          render: (r) => `${String(r.n_complete)} / ${String(r.n_trials)}`,
        },
        {
          header: "Best",
          align: "right",
          cellClassName: "font-mono",
          render: (r) => formatMetric(r.best_value),
          sortKey: "best_value",
        },
        {
          header: "Created",
          cellClassName: "font-mono text-xs",
          render: (r) => formatDateTime(r.created_at),
          sortKey: "created_at",
        },
      ]}
    />
  );
}
