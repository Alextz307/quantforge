import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useHpoStudies, usePrefetchHpoStudy, type HpoSummary } from "@/api/hpo";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { hpoDetailPath } from "@/lib/routes";

function applyFilters(rows: readonly HpoSummary[], store: string, since: string): HpoSummary[] {
  const sinceMs = since ? new Date(since).getTime() : null;
  return rows.filter((r) => {
    if (store !== ALL_OPTION && r.store !== store) return false;
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
  const filtered = useMemo(() => applyFilters(rows, store, since), [rows, store, since]);
  const prefetchHpo = usePrefetchHpoStudy();

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FilterField id="filter-store" label="Store">
          <select
            id="filter-store"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={store}
            onChange={(e) => {
              onStore(e.target.value);
            }}
          >
            <option value={ALL_OPTION}>All stores</option>
            {stores.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </FilterField>
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
      </div>

      {filtered.length === 0 ? (
        <p className="text-sm text-muted-foreground">No HPO studies match the current filters.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="hpo-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Store</th>
                <th className="py-2 pr-4">Direction</th>
                <th className="py-2 pr-4 text-right">Trials</th>
                <th className="py-2 pr-4 text-right">Best</th>
                <th className="py-2 pr-0">Created</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr
                  key={r.name}
                  className="border-b last:border-0"
                  onMouseEnter={() => {
                    prefetchHpo(r.name);
                  }}
                >
                  <td className="py-2 pr-4">
                    <Link to={hpoDetailPath(r.name)} className="text-primary hover:underline">
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.store}</td>
                  <td className="py-2 pr-4 font-mono">{r.direction}</td>
                  <td className="py-2 pr-4 text-right font-mono">
                    {r.n_complete} / {r.n_trials}
                  </td>
                  <td className="py-2 pr-4 text-right font-mono">{formatMetric(r.best_value)}</td>
                  <td className="py-2 pr-0 font-mono text-xs">{formatDateTime(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
