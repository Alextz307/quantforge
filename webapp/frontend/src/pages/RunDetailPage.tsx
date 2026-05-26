import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { plotDownloadUrl, useRun, useRunFolds, type FoldRow } from "@/api/runs";
import { BackLink } from "@/components/BackLink";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityChart, type EquityTrace } from "@/components/charts/EquityChart";
import { FoldMetricsTable } from "@/components/runs/FoldMetricsTable";
import { ManifestPanel } from "@/components/runs/ManifestPanel";
import { PlotIndex } from "@/components/PlotIndex";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatMetric } from "@/lib/format";
import { ROUTES } from "@/lib/routes";
import { SOURCE_KIND_RUN } from "@/lib/sourceKind";

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

function foldsToTraces(folds: readonly FoldRow[]): EquityTrace[] {
  return folds.map((f) => ({ name: `Fold ${String(f.fold_index)}`, equity: f.equity_curve }));
}

export function RunDetailPage() {
  const { experimentId = "" } = useParams<{ experimentId: string }>();
  const runQuery = useRun(experimentId);
  const foldsQuery = useRunFolds(experimentId);

  return (
    <QueryRenderer query={runQuery} errorTitle="Failed to load run" loadingMessage="Loading run…">
      {(run) => (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <BackLink to={ROUTES.runs}>All runs</BackLink>
            {run.holdout_start !== null && (
              <Button asChild variant="outline" size="sm">
                <Link
                  to={`${ROUTES.configureHoldout}?source_kind=${SOURCE_KIND_RUN}&source_id=${encodeURIComponent(run.experiment_id)}`}
                  data-testid="run-detail-holdout-cta"
                >
                  Run holdout eval
                </Link>
              </Button>
            )}
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
                {(folds) => <EquityChart traces={foldsToTraces(folds)} xLabel="Bar within fold" />}
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
              <PlotIndex
                plots={run.plots}
                urlForPlot={(name) => plotDownloadUrl(run.experiment_id, name)}
                emptyMessage="No static figures produced for this run."
              />
            </CardContent>
          </Card>
        </div>
      )}
    </QueryRenderer>
  );
}
