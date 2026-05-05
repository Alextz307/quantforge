import { useMemo } from "react";
import { useParams } from "react-router-dom";
import { plotDownloadUrl, useRun, useRunFolds, type FoldRow } from "@/api/runs";
import { BackLink } from "@/components/BackLink";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityChart, type EquityTrace } from "@/components/charts/EquityChart";
import { FoldMetricsTable } from "@/components/runs/FoldMetricsTable";
import { ManifestPanel } from "@/components/runs/ManifestPanel";
import { PlotIndex } from "@/components/PlotIndex";
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

function foldsToTraces(folds: readonly FoldRow[]): EquityTrace[] {
  return folds.map((f) => ({ name: `Fold ${String(f.fold_index)}`, equity: f.equity_curve }));
}

function nestingKind(store: string): "comparison" | "hpo" | null {
  const segs = store.split("/");
  if (segs.includes("comparisons")) return "comparison";
  if (segs.includes("hpo")) return "hpo";
  return null;
}

function emptyPlotsMessage(store: string): string {
  const kind = nestingKind(store);
  if (kind === "comparison") {
    return "No static figures — this run is nested inside a comparison; the comparison renders its own figures. Re-run via `experiment run` for per-run figures.";
  }
  if (kind === "hpo") {
    return "No static figures — this run is an individual HPO trial. Per-trial figures are skipped during the sweep; the HPO study page shows convergence and best-trial views.";
  }
  return "No static figures produced for this run.";
}

export function RunDetailPage() {
  const { experimentId = "" } = useParams<{ experimentId: string }>();
  const runQuery = useRun(experimentId);
  const foldsQuery = useRunFolds(experimentId);

  return (
    <QueryRenderer query={runQuery} errorTitle="Failed to load run" loadingMessage="Loading run…">
      {(run) => (
        <div className="flex flex-col gap-4">
          <BackLink to={ROUTES.runs}>All runs</BackLink>
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
                emptyMessage={emptyPlotsMessage(run.store)}
              />
            </CardContent>
          </Card>
        </div>
      )}
    </QueryRenderer>
  );
}
