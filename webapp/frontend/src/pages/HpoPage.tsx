import { useMemo, useState } from "react";
import { useMe } from "@/api/auth";
import {
  useHpoStudiesPage,
  usePrefetchHpoStudy,
  type HpoSortBy,
  type HpoStudiesPage,
  type HpoSummary,
} from "@/api/hpo";
import { AllUsersToggle } from "@/components/AllUsersToggle";
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
import {
  ALL_OPTION,
  readSortState,
  readValidSince,
  toggleSortParams,
  withActiveOption,
} from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { hpoDetailPath } from "@/lib/routes";

const DEFAULT_SORT: SortState<HpoSortBy> = { sortBy: "created_at", order: "desc" };
const SORT_KEYS: ReadonlySet<HpoSortBy> = new Set(["created_at", "best_value"]);

interface HpoUrlState {
  sortBy: HpoSortBy;
  order: SortOrder;
  store: string;
  since: string;
}

function readState(params: URLSearchParams): HpoUrlState {
  return {
    ...readSortState(params, SORT_KEYS, DEFAULT_SORT),
    store: params.get("store") ?? ALL_OPTION,
    since: readValidSince(params.get("since")),
  };
}

export function HpoPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const { searchParams, limit, offset, setOffset, setParams } = usePaginatedSearch();
  const urlState = useMemo(() => readState(searchParams), [searchParams]);
  const sortState = useMemo<SortState<HpoSortBy>>(
    () => ({ sortBy: urlState.sortBy, order: urlState.order }),
    [urlState.sortBy, urlState.order],
  );

  const query = useHpoStudiesPage(
    {
      limit,
      offset,
      sortBy: urlState.sortBy,
      order: urlState.order,
      ...(urlState.store !== ALL_OPTION ? { store: urlState.store } : {}),
      ...(urlState.since ? { since: new Date(urlState.since).toISOString() } : {}),
    },
    { allUsers: isAdmin && allUsers },
  );

  const onSortToggle = (col: HpoSortBy) => {
    setParams(toggleSortParams(sortState, col));
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>HPO studies</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="HPO studies"
          testId="hpo-all-users-toggle"
        />
        <QueryRenderer query={query} errorTitle="Failed to load HPO studies">
          {(page) => (
            <HpoBody
              page={page}
              store={urlState.store}
              since={urlState.since}
              onStore={(v) => {
                setParams({ store: v === ALL_OPTION ? "" : v });
              }}
              onSince={(v) => {
                setParams({ since: v });
              }}
              sortState={sortState}
              onSortToggle={onSortToggle}
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
  page: HpoStudiesPage;
  store: string;
  since: string;
  onStore: (v: string) => void;
  onSince: (v: string) => void;
  sortState: SortState<HpoSortBy>;
  onSortToggle: (col: HpoSortBy) => void;
  limit: number;
  onOffset: (offset: number) => void;
}

function HpoBody({
  page,
  store,
  since,
  onStore,
  onSince,
  sortState,
  onSortToggle,
  limit,
  onOffset,
}: BodyProps) {
  const storeOptions = useMemo(() => withActiveOption(page.stores, store), [page.stores, store]);
  const prefetchHpo = usePrefetchHpoStudy();

  return (
    <div className="flex flex-col gap-4">
      <FilterableTablePage<HpoSummary, Record<string, never>, HpoSortBy>
        rows={page.items}
        filterControls={
          <>
            <FilterSelect
              id="filter-store"
              label="Store"
              value={store}
              onChange={onStore}
              allLabel="All stores"
              options={storeOptions}
            />
            <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
          </>
        }
        rowKey={(r) => r.wire_id}
        rowName={(r) => r.name}
        rowHref={(r) => hpoDetailPath(r.wire_id)}
        rowOnHover={(r) => {
          prefetchHpo(r.wire_id);
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
