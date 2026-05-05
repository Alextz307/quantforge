import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useComparisons, usePrefetchComparison, type ComparisonSummary } from "@/api/comparisons";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { comparisonDetailPath } from "@/lib/routes";

function applyFilters(
  rows: readonly ComparisonSummary[],
  strategy: string,
  since: string,
): ComparisonSummary[] {
  const sinceMs = since ? new Date(since).getTime() : null;
  return rows.filter((r) => {
    if (strategy !== ALL_OPTION && !r.strategies.includes(strategy)) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function ComparisonsPage() {
  const query = useComparisons();
  const [strategy, setStrategy] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Comparisons</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
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
  const filtered = useMemo(() => applyFilters(rows, strategy, since), [rows, strategy, since]);
  const prefetchComparison = usePrefetchComparison();

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FilterField id="filter-strategy" label="Strategy">
          <select
            id="filter-strategy"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={strategy}
            onChange={(e) => {
              onStrategy(e.target.value);
            }}
          >
            <option value={ALL_OPTION}>All strategies</option>
            {strategies.map((s) => (
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
        <p className="text-sm text-muted-foreground">No comparisons match the current filters.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="comparisons-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Store</th>
                <th className="py-2 pr-4">Strategies</th>
                <th className="py-2 pr-0">Created</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr
                  key={r.name}
                  className="border-b last:border-0"
                  onMouseEnter={() => {
                    prefetchComparison(r.name);
                  }}
                >
                  <td className="py-2 pr-4">
                    <Link
                      to={comparisonDetailPath(r.name)}
                      className="text-primary hover:underline"
                    >
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.store}</td>
                  <td className="py-2 pr-4 font-mono">{r.strategies.join(", ")}</td>
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
