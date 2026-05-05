import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useHoldoutEvals, type HoldoutEvalSummary } from "@/api/holdout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { holdoutDetailPath } from "@/lib/routes";

function applyFilters(
  rows: readonly HoldoutEvalSummary[],
  sourceKind: string,
  since: string,
): HoldoutEvalSummary[] {
  const sinceMs = since ? new Date(since).getTime() : null;
  return rows.filter((r) => {
    if (sourceKind !== ALL_OPTION && r.source_kind !== sourceKind) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function HoldoutPage() {
  const query = useHoldoutEvals();
  const [sourceKind, setSourceKind] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Holdout evaluations</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <QueryRenderer query={query} errorTitle="Failed to load holdout evaluations">
          {(rows) => (
            <HoldoutBody
              rows={rows}
              sourceKind={sourceKind}
              since={since}
              onSourceKind={setSourceKind}
              onSince={setSince}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly HoldoutEvalSummary[];
  sourceKind: string;
  since: string;
  onSourceKind: (v: string) => void;
  onSince: (v: string) => void;
}

function HoldoutBody({ rows, sourceKind, since, onSourceKind, onSince }: BodyProps) {
  const sourceKinds = useMemo(() => uniqSorted(rows.map((r) => r.source_kind)), [rows]);
  const filtered = useMemo(() => applyFilters(rows, sourceKind, since), [rows, sourceKind, since]);

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FilterField id="filter-source-kind" label="Source kind">
          <select
            id="filter-source-kind"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={sourceKind}
            onChange={(e) => {
              onSourceKind(e.target.value);
            }}
          >
            <option value={ALL_OPTION}>All source kinds</option>
            {sourceKinds.map((k) => (
              <option key={k} value={k}>
                {k}
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
        <p className="text-sm text-muted-foreground">
          No holdout evaluations match the current filters.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="holdout-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Store</th>
                <th className="py-2 pr-4">Source</th>
                <th className="py-2 pr-4">Holdout start</th>
                <th className="py-2 pr-0">Created</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.name} className="border-b last:border-0">
                  <td className="py-2 pr-4">
                    <Link to={holdoutDetailPath(r.name)} className="text-primary hover:underline">
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.store}</td>
                  <td className="py-2 pr-4 font-mono text-xs">
                    {r.source_kind} · {r.source_id}
                  </td>
                  <td className="py-2 pr-4 font-mono text-xs">{formatDateTime(r.holdout_start)}</td>
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
