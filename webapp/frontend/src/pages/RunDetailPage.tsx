import { memo, useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useCreateDeployment } from "@/api/deployments";
import {
  plotDownloadUrl,
  useFeatureImportance,
  useRun,
  useRunFolds,
  type FoldRow,
} from "@/api/runs";
import { BackLink } from "@/components/BackLink";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityChart, type EquityTrace } from "@/components/charts/EquityChart";
import { FeatureImportanceView } from "@/components/runs/FeatureImportanceView";
import { FoldMetricsTable } from "@/components/runs/FoldMetricsTable";
import { ManifestPanel } from "@/components/runs/ManifestPanel";
import { PlotIndex } from "@/components/PlotIndex";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatMetric } from "@/lib/format";
import { deploymentDetailPath, ROUTES } from "@/lib/routes";
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

// Memoized on the folds reference so a background job finishing (which re-renders
// RunDetailPage via the importance query) doesn't reflow the untouched equity chart.
const EquitySection = memo(function EquitySection({ folds }: { folds: readonly FoldRow[] }) {
  return <EquityChart traces={foldsToTraces(folds)} xLabel="Bar within fold" />;
});

export function RunDetailPage() {
  const { experimentId = "" } = useParams<{ experimentId: string }>();
  // Remount per run id: a client-side nav between run detail pages otherwise
  // reuses this instance with new data, and Plotly keeps a stale size and
  // overlaps until reload. Keying by id gives each run a clean mount.
  return <RunDetailContent key={experimentId} experimentId={experimentId} />;
}

function RunDetailContent({ experimentId }: { experimentId: string }) {
  const runQuery = useRun(experimentId);
  const foldsQuery = useRunFolds(experimentId);
  const importanceQuery = useFeatureImportance(experimentId);
  const create = useCreateDeployment();
  const navigate = useNavigate();

  function deploy() {
    create.mutate(
      { source_kind: SOURCE_KIND_RUN, source_id: experimentId },
      {
        onSuccess: (d) => {
          navigate(deploymentDetailPath(d.id));
        },
      },
    );
  }

  return (
    <QueryRenderer query={runQuery} errorTitle="Failed to load run" loadingMessage="Loading run...">
      {(run) => (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <BackLink to={ROUTES.runs}>All runs</BackLink>
            <div className="flex items-center gap-2">
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
              <Button
                size="sm"
                disabled={create.isPending}
                onClick={deploy}
                data-testid="run-detail-deploy-cta"
              >
                Deploy
              </Button>
            </div>
          </div>
          {create.isError && (
            <Alert variant="destructive">
              <AlertDescription>{create.error.message}</AlertDescription>
            </Alert>
          )}
          {run.holdout_start === null && (
            <Alert>
              <AlertTitle>No holdout evaluation</AlertTitle>
              <AlertDescription>
                This run has no holdout evaluation; deploying without out-of-sample validation is
                possible but not recommended.
              </AlertDescription>
            </Alert>
          )}
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
                loadingMessage="Loading folds..."
              >
                {(folds) => <EquitySection folds={folds} />}
              </QueryRenderer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Feature importance</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-3 text-sm text-muted-foreground">
                Out-of-sample permutation importance (mean score drop &plusmn; across-fold std);
                XGBoost gain where available. The ARMA/GARCH+LSTM hybrids route features only
                through the residual correction, so their bars compress toward zero.
              </p>
              <QueryRenderer
                query={importanceQuery}
                errorTitle="Failed to load feature importance"
                loadingMessage="Loading feature importance..."
              >
                {(importance) => (
                  <FeatureImportanceView
                    experimentId={experimentId}
                    strategyName={run.strategy}
                    response={importance}
                  />
                )}
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
                loadingMessage="Loading folds..."
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
