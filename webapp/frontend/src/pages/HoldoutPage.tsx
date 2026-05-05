import { useMemo, useState } from "react";
import { useHoldoutEvals, usePrefetchHoldoutEval, type HoldoutEvalSummary } from "@/api/holdout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime } from "@/lib/format";
import { holdoutDetailPath } from "@/lib/routes";
import { SOURCE_KINDS, sourceKindLabel, type SourceKind } from "@/lib/sourceKind";

type SourceKindFilter = SourceKind | typeof ALL_OPTION;

function isSourceKindFilter(value: string): value is SourceKindFilter {
  return value === ALL_OPTION || (SOURCE_KINDS as readonly string[]).includes(value);
}

interface HoldoutFilters {
  sourceKind: SourceKindFilter;
  since: string;
}

function applyFilters(
  rows: readonly HoldoutEvalSummary[],
  f: HoldoutFilters,
): readonly HoldoutEvalSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.sourceKind !== ALL_OPTION && r.source_kind !== f.sourceKind) return false;
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
  const filters = useMemo<HoldoutFilters>(() => ({ sourceKind, since }), [sourceKind, since]);
  const prefetchHoldoutEval = usePrefetchHoldoutEval();

  return (
    <FilterableTablePage<HoldoutEvalSummary, HoldoutFilters>
      rows={rows}
      filters={filters}
      applyFilters={applyFilters}
      filterControls={
        <>
          <FilterSelect
            id="filter-source-kind"
            label="Source kind"
            value={sourceKind}
            onChange={(next) => {
              if (isSourceKindFilter(next)) onSourceKind(next);
            }}
            allLabel="All source kinds"
            options={sourceKindOptions}
            optionLabel={(v) => sourceKindLabel(v as SourceKind)}
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
      rowHref={(r) => holdoutDetailPath(r.name)}
      rowOnHover={(r) => {
        prefetchHoldoutEval(r.name);
      }}
      tableTestId="holdout-table"
      emptyMessage="No holdout evaluations match the current filters."
      columns={[
        { header: "Store", cellClassName: "font-mono", render: (r) => r.store },
        {
          header: "Source",
          cellClassName: "font-mono text-xs",
          render: (r) => `${sourceKindLabel(r.source_kind)} · ${r.source_id}`,
        },
        {
          header: "Holdout start",
          cellClassName: "font-mono text-xs",
          render: (r) => formatDateTime(r.holdout_start),
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
