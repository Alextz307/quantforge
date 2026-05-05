import { useMemo, useState } from "react";
import { useHpoStudies, usePrefetchHpoStudy, type HpoSummary } from "@/api/hpo";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { hpoDetailPath } from "@/lib/routes";

interface HpoFilters {
  store: string;
  since: string;
}

function applyFilters(rows: readonly HpoSummary[], f: HpoFilters): readonly HpoSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.store !== ALL_OPTION && r.store !== f.store) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function HpoPage() {
  const query = useHpoStudies();
  const [store, setStore] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

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
}

function HpoBody({ rows, store, since, onStore, onSince }: BodyProps) {
  const stores = useMemo(() => uniqSorted(rows.map((r) => r.store)), [rows]);
  const filters = useMemo<HpoFilters>(() => ({ store, since }), [store, since]);
  const prefetchHpo = usePrefetchHpoStudy();

  return (
    <FilterableTablePage<HpoSummary, HpoFilters>
      rows={rows}
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
          <FilterField id="filter-since" label="Since">
            <Input
              id="filter-since"
              type="date"
              value={since}
              onChange={(e) => {
                onSince(e.target.value);
              }}
            />
          </FilterField>
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
        },
        {
          header: "Created",
          cellClassName: "font-mono text-xs",
          render: (r) => formatDateTime(r.created_at),
        },
      ]}
    />
  );
}
