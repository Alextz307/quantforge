import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  useHoldoutEvals,
  usePrefetchHoldoutEval,
  type HoldoutEvalSummary,
} from "@/api/holdout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { holdoutDetailPath } from "@/lib/routes";
import {
  SOURCE_KINDS,
  sourceKindLabel,
  type SourceKind,
} from "@/lib/sourceKind";

type SourceKindFilter = SourceKind | typeof ALL_OPTION;

function isSourceKindFilter(value: string): value is SourceKindFilter {
  return value === ALL_OPTION || (SOURCE_KINDS as readonly string[]).includes(value);
}

function applyFilters(
  rows: readonly HoldoutEvalSummary[],
  sourceKind: SourceKindFilter,
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
  const [sourceKind, setSourceKind] = useState<SourceKindFilter>(ALL_OPTION);
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
  sourceKind: SourceKindFilter;
  since: string;
  onSourceKind: (v: SourceKindFilter) => void;
  onSince: (v: string) => void;
}

function HoldoutBody({ rows, sourceKind, since, onSourceKind, onSince }: BodyProps) {
  const sourceKindOptions = useMemo<SourceKind[]>(
    () => uniqSorted(rows.map((r) => r.source_kind)) as SourceKind[],
    [rows],
  );
  const filtered = useMemo(() => applyFilters(rows, sourceKind, since), [rows, sourceKind, since]);
  const prefetchHoldoutEval = usePrefetchHoldoutEval();

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FilterField id="filter-source-kind" label="Source kind">
          <select
            id="filter-source-kind"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={sourceKind}
            onChange={(e) => {
              const next = e.target.value;
              if (isSourceKindFilter(next)) onSourceKind(next);
            }}
          >
            <option value={ALL_OPTION}>All source kinds</option>
            {sourceKindOptions.map((k) => (
              <option key={k} value={k}>
                {sourceKindLabel(k)}
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
                <tr
                  key={r.name}
                  className="border-b last:border-0"
                  onMouseEnter={() => {
                    prefetchHoldoutEval(r.name);
                  }}
                >
                  <td className="py-2 pr-4">
                    <Link to={holdoutDetailPath(r.name)} className="text-primary hover:underline">
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.store}</td>
                  <td className="py-2 pr-4 font-mono text-xs">
                    {sourceKindLabel(r.source_kind)} · {r.source_id}
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
