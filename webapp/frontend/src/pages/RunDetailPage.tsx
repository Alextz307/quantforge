import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { useRun, useRunFolds } from "@/api/runs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityChart } from "@/components/charts/EquityChart";
import { FoldMetricsTable } from "@/components/runs/FoldMetricsTable";
import { ManifestPanel } from "@/components/runs/ManifestPanel";
import { PlotIndex } from "@/components/runs/PlotIndex";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatMetric } from "@/lib/format";
import { ROUTES } from "@/lib/routes";

function MetricsGrid({ metrics }: { metrics: Record<string, number> }) {
  const entries = useMemo(
    () => Object.entries(metrics).sort(([a], [b]) => a.localeCompare(b)),
    [metrics],
  );
  if (entries.length === 0) {
    return <p className="text-sm text-muted-foreground">No aggregated metrics recorded.</p>;
  }
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4" data-testid="metrics-grid">
      {entries.map(([k, v]) => (
        <div key={k} className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{k}</span>
          <span className="text-sm font-mono">{formatMetric(v)}</span>
        </div>
      ))}
    </div>
  );
}

export function RunDetailPage() {
  const { experimentId = "" } = useParams<{ experimentId: string }>();
  const runQuery = useRun(experimentId);
  const foldsQuery = useRunFolds(experimentId);

  return (
    <QueryRenderer query={runQuery} errorTitle="Failed to load run" loadingMessage="Loading run…">
      {(run) => (
        <div className="flex flex-col gap-4">
          <div>
            <Link to={ROUTES.runs} className="text-xs text-primary hover:underline">
              ← All runs
            </Link>
          </div>
          <ManifestPanel run={run} />

          <Card>
            <CardHeader>
              <CardTitle>Aggregated metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <MetricsGrid metrics={run.metrics} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Equity curves</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryRenderer
                query={foldsQuery}
                errorTitle="Failed to load folds"
                loadingMessage="Loading folds…"
              >
                {(folds) => <EquityChart folds={folds} />}
              </QueryRenderer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Per-fold metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryRenderer
                query={foldsQuery}
                errorTitle="Failed to load folds"
                loadingMessage="Loading folds…"
              >
                {(folds) => <FoldMetricsTable folds={folds} />}
              </QueryRenderer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Static figures</CardTitle>
            </CardHeader>
            <CardContent>
              <PlotIndex experimentId={run.experiment_id} plots={run.plots} />
            </CardContent>
          </Card>
        </div>
      )}
    </QueryRenderer>
  );
}
