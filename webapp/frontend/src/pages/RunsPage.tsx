import { useMemo, useState } from "react";
import { usePrefetchRun, useRuns, type RunSummary } from "@/api/runs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";

interface RunsFilters {
  strategy: string;
  ticker: string;
  since: string;
}

function applyFilters(runs: readonly RunSummary[], f: RunsFilters): readonly RunSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return runs.filter((r) => {
    if (f.strategy !== ALL_OPTION && r.strategy !== f.strategy) return false;
    if (f.ticker !== ALL_OPTION && !r.tickers.includes(f.ticker)) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

export function RunsPage() {
  const runsQuery = useRuns();
  const [strategy, setStrategy] = useState<string>(ALL_OPTION);
  const [ticker, setTicker] = useState<string>(ALL_OPTION);
  const [since, setSince] = useState<string>("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <QueryRenderer query={runsQuery} errorTitle="Failed to load runs">
          {(allRuns) => (
            <RunsBody
              runs={allRuns}
              strategy={strategy}
              ticker={ticker}
              since={since}
              onStrategy={setStrategy}
              onTicker={setTicker}
              onSince={setSince}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface RunsBodyProps {
  runs: readonly RunSummary[];
  strategy: string;
  ticker: string;
  since: string;
  onStrategy: (v: string) => void;
  onTicker: (v: string) => void;
  onSince: (v: string) => void;
}

function RunsBody({ runs, strategy, ticker, since, onStrategy, onTicker, onSince }: RunsBodyProps) {
  const strategies = useMemo(() => uniqSorted(runs.map((r) => r.strategy)), [runs]);
  const tickers = useMemo(() => uniqSorted(runs.flatMap((r) => r.tickers)), [runs]);
  const filters = useMemo<RunsFilters>(
    () => ({ strategy, ticker, since }),
    [strategy, ticker, since],
  );
  const prefetchRun = usePrefetchRun();

  return (
    <FilterableTablePage<RunSummary, RunsFilters>
      rows={runs}
      filters={filters}
      applyFilters={applyFilters}
      filterGridClassName="md:grid-cols-3"
      filterControls={
        <>
          <FilterSelect
            id="filter-strategy"
            label="Strategy"
            value={strategy}
            onChange={onStrategy}
            allLabel="All strategies"
            options={strategies}
          />
          <FilterSelect
            id="filter-ticker"
            label="Ticker"
            value={ticker}
            onChange={onTicker}
            allLabel="All tickers"
            options={tickers}
          />
          <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
        </>
      }
      rowKey={(r) => r.experiment_id}
      rowName={(r) => r.name}
      rowHref={(r) => runDetailPath(r.experiment_id)}
      rowOnHover={(r) => {
        prefetchRun(r.experiment_id);
      }}
      tableTestId="runs-table"
      emptyMessage="No runs match the current filters."
      columns={[
        { header: "Strategy", cellClassName: "font-mono", render: (r) => r.strategy },
        { header: "Tickers", cellClassName: "font-mono", render: (r) => r.tickers.join(", ") },
        { header: "Interval", cellClassName: "font-mono", render: (r) => r.interval },
        {
          header: "Created",
          cellClassName: "font-mono text-xs",
          render: (r) => formatDateTime(r.created_at),
        },
        {
          header: "Sharpe",
          align: "right",
          cellClassName: "font-mono",
          render: (r) => formatMetric(r.sharpe_mean, 3),
        },
        {
          header: "Calmar",
          align: "right",
          cellClassName: "font-mono",
          render: (r) => formatMetric(r.calmar_mean, 3),
        },
      ]}
    />
  );
}
