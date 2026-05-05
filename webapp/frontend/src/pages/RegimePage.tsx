import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useRegimeReports, usePrefetchRegimeReport, type RegimeReportSummary } from "@/api/regime";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { regimeDetailPath } from "@/lib/routes";

function applyFilters(
  rows: readonly RegimeReportSummary[],
  detector: string,
  since: string,
): RegimeReportSummary[] {
  const sinceMs = since ? new Date(since).getTime() : null;
  return rows.filter((r) => {
    if (detector !== ALL_OPTION && r.detector_name !== detector) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function RegimePage() {
  const query = useRegimeReports();
  const [detector, setDetector] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Regime reports</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <QueryRenderer query={query} errorTitle="Failed to load regime reports">
          {(rows) => (
            <RegimeBody
              rows={rows}
              detector={detector}
              since={since}
              onDetector={setDetector}
              onSince={setSince}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly RegimeReportSummary[];
  detector: string;
  since: string;
  onDetector: (v: string) => void;
  onSince: (v: string) => void;
}

function RegimeBody({ rows, detector, since, onDetector, onSince }: BodyProps) {
  const detectors = useMemo(() => uniqSorted(rows.map((r) => r.detector_name)), [rows]);
  const filtered = useMemo(() => applyFilters(rows, detector, since), [rows, detector, since]);
  const prefetchRegime = usePrefetchRegimeReport();

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FilterField id="filter-detector" label="Detector">
          <select
            id="filter-detector"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={detector}
            onChange={(e) => {
              onDetector(e.target.value);
            }}
          >
            <option value={ALL_OPTION}>All detectors</option>
            {detectors.map((d) => (
              <option key={d} value={d}>
                {d}
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
          No regime reports match the current filters.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="regime-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Store</th>
                <th className="py-2 pr-4">Detector</th>
                <th className="py-2 pr-4">Regimes</th>
                <th className="py-2 pr-0">Created</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr
                  key={r.name}
                  className="border-b last:border-0"
                  onMouseEnter={() => {
                    prefetchRegime(r.name);
                  }}
                >
                  <td className="py-2 pr-4">
                    <Link to={regimeDetailPath(r.name)} className="text-primary hover:underline">
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.store}</td>
                  <td className="py-2 pr-4 font-mono">{r.detector_name}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{r.regime_labels.join(", ")}</td>
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
