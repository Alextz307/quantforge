import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useStudies, usePrefetchStudy, type StudySummary } from "@/api/studies";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatPercent } from "@/lib/format";
import { studyDetailPath } from "@/lib/routes";

function applyFilters(
  rows: readonly StudySummary[],
  spec: string,
  since: string,
): StudySummary[] {
  const sinceMs = since ? new Date(since).getTime() : null;
  return rows.filter((r) => {
    if (spec !== ALL_OPTION && r.spec_name !== spec) return false;
    if (sinceMs != null && new Date(r.started_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function StudiesPage() {
  const query = useStudies();
  const [spec, setSpec] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Studies</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <QueryRenderer query={query} errorTitle="Failed to load studies">
          {(rows) => (
            <StudiesBody
              rows={rows}
              spec={spec}
              since={since}
              onSpec={setSpec}
              onSince={setSince}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly StudySummary[];
  spec: string;
  since: string;
  onSpec: (v: string) => void;
  onSince: (v: string) => void;
}

function StudiesBody({ rows, spec, since, onSpec, onSince }: BodyProps) {
  const specs = useMemo(() => uniqSorted(rows.map((r) => r.spec_name)), [rows]);
  const filtered = useMemo(() => applyFilters(rows, spec, since), [rows, spec, since]);
  const prefetchStudy = usePrefetchStudy();

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FilterField id="filter-spec" label="Spec">
          <select
            id="filter-spec"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={spec}
            onChange={(e) => {
              onSpec(e.target.value);
            }}
          >
            <option value={ALL_OPTION}>All specs</option>
            {specs.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </FilterField>
        <FilterField id="filter-since" label="Started since">
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
        <p className="text-sm text-muted-foreground">No studies match the current filters.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="studies-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Spec</th>
                <th className="py-2 pr-4 text-right">Legs</th>
                <th className="py-2 pr-4 text-right">Completion</th>
                <th className="py-2 pr-0">Started</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr
                  key={r.name}
                  className="border-b last:border-0"
                  onMouseEnter={() => {
                    prefetchStudy(r.name);
                  }}
                >
                  <td className="py-2 pr-4">
                    <Link to={studyDetailPath(r.name)} className="text-primary hover:underline">
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.spec_name}</td>
                  <td className="py-2 pr-4 text-right font-mono">
                    {r.completed_legs} / {r.total_legs}
                  </td>
                  <td className="py-2 pr-4 text-right font-mono">
                    {formatPercent(r.completion_pct / 100)}
                  </td>
                  <td className="py-2 pr-0 font-mono text-xs">{formatDateTime(r.started_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
