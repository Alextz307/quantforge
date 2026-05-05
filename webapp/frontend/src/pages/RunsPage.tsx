import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useRuns, type RunSummary } from "@/api/runs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, formatMetric } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";

const ALL_OPTION = "__all__";

function uniqSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values)).sort();
}

function applyFilters(
  runs: readonly RunSummary[],
  strategy: string,
  ticker: string,
  since: string,
): RunSummary[] {
  const sinceMs = since ? new Date(since).getTime() : null;
  return runs.filter((r) => {
    if (strategy !== ALL_OPTION && r.strategy !== strategy) return false;
    if (ticker !== ALL_OPTION && !r.tickers.includes(ticker)) return false;
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
  const filtered = useMemo(
    () => applyFilters(runs, strategy, ticker, since),
    [runs, strategy, ticker, since],
  );

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
        <FilterField id="filter-ticker" label="Ticker">
          <select
            id="filter-ticker"
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={ticker}
            onChange={(e) => {
              onTicker(e.target.value);
            }}
          >
            <option value={ALL_OPTION}>All tickers</option>
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
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
        <p className="text-sm text-muted-foreground">No runs match the current filters.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="runs-table">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Strategy</th>
                <th className="py-2 pr-4">Tickers</th>
                <th className="py-2 pr-4">Interval</th>
                <th className="py-2 pr-4">Created</th>
                <th className="py-2 pr-4 text-right">Sharpe</th>
                <th className="py-2 pr-0 text-right">Calmar</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.experiment_id} className="border-b last:border-0">
                  <td className="py-2 pr-4">
                    <Link
                      to={runDetailPath(r.experiment_id)}
                      className="text-primary hover:underline"
                    >
                      {r.name}
                    </Link>
                  </td>
                  <td className="py-2 pr-4 font-mono">{r.strategy}</td>
                  <td className="py-2 pr-4 font-mono">{r.tickers.join(", ")}</td>
                  <td className="py-2 pr-4 font-mono">{r.interval}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{formatDateTime(r.created_at)}</td>
                  <td className="py-2 pr-4 text-right font-mono">
                    {formatMetric(r.sharpe_mean, 3)}
                  </td>
                  <td className="py-2 pr-0 text-right font-mono">
                    {formatMetric(r.calmar_mean, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
