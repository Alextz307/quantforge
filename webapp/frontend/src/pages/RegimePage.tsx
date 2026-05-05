import { useMemo, useState } from "react";
import { useRegimeReports, usePrefetchRegimeReport, type RegimeReportSummary } from "@/api/regime";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { regimeDetailPath } from "@/lib/routes";

interface RegimeFilters {
  detector: string;
  since: string;
}

function applyFilters(
  rows: readonly RegimeReportSummary[],
  f: RegimeFilters,
): readonly RegimeReportSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.detector !== ALL_OPTION && r.detector_name !== f.detector) return false;
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
  const filters = useMemo<RegimeFilters>(() => ({ detector, since }), [detector, since]);
  const prefetchRegime = usePrefetchRegimeReport();

  return (
    <FilterableTablePage<RegimeReportSummary, RegimeFilters>
      rows={rows}
      filters={filters}
      applyFilters={applyFilters}
      filterControls={
        <>
          <FilterSelect
            id="filter-detector"
            label="Detector"
            value={detector}
            onChange={onDetector}
            allLabel="All detectors"
            options={detectors}
          />
          <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
        </>
      }
      rowKey={(r) => r.name}
      rowName={(r) => r.name}
      rowHref={(r) => regimeDetailPath(r.name)}
      rowOnHover={(r) => {
        prefetchRegime(r.name);
      }}
      tableTestId="regime-table"
      emptyMessage="No regime reports match the current filters."
      columns={[
        { header: "Store", cellClassName: "font-mono", render: (r) => r.store },
        { header: "Detector", cellClassName: "font-mono", render: (r) => r.detector_name },
        {
          header: "Regimes",
          cellClassName: "font-mono text-xs",
          render: (r) => r.regime_labels.join(", "),
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
