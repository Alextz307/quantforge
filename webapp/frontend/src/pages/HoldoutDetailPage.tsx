import { Link, useParams } from "react-router-dom";
import { holdoutPlotDownloadUrl, useHoldoutEval, type HoldoutEvalDetail } from "@/api/holdout";
import { BackLink } from "@/components/BackLink";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityChart, type EquityTrace } from "@/components/charts/EquityChart";
import { DevVsHoldoutPanel } from "@/components/holdout/DevVsHoldoutPanel";
import { MetadataField } from "@/components/MetadataField";
import { PlotIndex } from "@/components/PlotIndex";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, shortHash } from "@/lib/format";
import { holdoutDetailPath, ROUTES, runDetailPath } from "@/lib/routes";
import { SOURCE_KIND_RUN } from "@/lib/sourceKind";

function IdentityCard({ holdout }: { holdout: HoldoutEvalDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{holdout.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetadataField label="Store" value={holdout.store} />
        <MetadataField label="Created" value={formatDateTime(holdout.created_at)} />
        <MetadataField label="Git SHA" value={shortHash(holdout.git_sha)} />
        <MetadataField label="Holdout start" value={formatDateTime(holdout.holdout_start)} />
        <MetadataField label="Data hash" value={shortHash(holdout.data_hash)} />
        <MetadataField label="Cost tier" value={holdout.slippage_scenario} />
      </CardContent>
    </Card>
  );
}

function holdoutToTraces(holdout: HoldoutEvalDetail): EquityTrace[] {
  if (holdout.equity_curve.length === 0) return [];
  return [{ name: "Holdout", equity: holdout.equity_curve }];
}

function SourceImportanceCard({ holdout }: { holdout: HoldoutEvalDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Feature importance</CardTitle>
      </CardHeader>
      <CardContent>
        {holdout.source_kind === SOURCE_KIND_RUN ? (
          <p className="text-sm text-muted-foreground">
            A holdout evaluation refits the source run's configuration on the full dev region and
            scores it once on the reserved window, so it has no per-fold importance of its own.
            Feature importance (computed across that run's walk-forward folds) lives on the source
            run, where you can also recompute it.{" "}
            <Link
              className="underline"
              to={runDetailPath(holdout.source_id)}
              state={{ from: holdoutDetailPath(holdout.name) }}
              data-testid="holdout-source-importance-link"
            >
              View the source run's feature importance &rarr;
            </Link>
          </p>
        ) : (
          <p className="text-sm text-muted-foreground" data-testid="holdout-source-importance-none">
            This evaluation derives from the HPO study{" "}
            <span className="font-mono">{holdout.source_id}</span>, whose trials don't compute
            feature importance. Run a backtest on the best config to inspect it.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function HoldoutDetailPage() {
  const { name = "" } = useParams<{ name: string }>();
  const query = useHoldoutEval(name);

  return (
    <QueryRenderer
      query={query}
      errorTitle="Failed to load holdout evaluation"
      loadingMessage="Loading holdout evaluation..."
    >
      {(holdout) => (
        <div className="flex flex-col gap-4">
          <BackLink to={ROUTES.holdout}>All holdout evaluations</BackLink>
          <IdentityCard holdout={holdout} />

          <DevVsHoldoutPanel holdout={holdout} />

          <SourceImportanceCard holdout={holdout} />

          <Card>
            <CardHeader>
              <CardTitle>Holdout equity curve</CardTitle>
            </CardHeader>
            <CardContent>
              <EquityChart
                traces={holdoutToTraces(holdout)}
                xLabel="Holdout bar"
                yLabel="Cumulative equity"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Static figures</CardTitle>
            </CardHeader>
            <CardContent>
              <PlotIndex
                plots={holdout.plots}
                urlForPlot={(plotName) => holdoutPlotDownloadUrl(holdout.name, plotName)}
                emptyMessage="No static figures produced for this evaluation."
              />
            </CardContent>
          </Card>
        </div>
      )}
    </QueryRenderer>
  );
}
