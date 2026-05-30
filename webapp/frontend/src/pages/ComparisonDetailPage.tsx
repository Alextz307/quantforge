import { useParams } from "react-router-dom";
import { comparisonPlotDownloadUrl, useComparison, type ComparisonDetail } from "@/api/comparisons";
import { BackLink } from "@/components/BackLink";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityOverlayChart, type OverlaySpec } from "@/components/charts/EquityOverlayChart";
import { MetadataField } from "@/components/MetadataField";
import { PerStrategyStatsTable } from "@/components/comparisons/PerStrategyStatsTable";
import { PlotIndex } from "@/components/PlotIndex";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, shortHash } from "@/lib/format";
import { ROUTES } from "@/lib/routes";

function IdentityCard({ comparison }: { comparison: ComparisonDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{comparison.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetadataField label="Store" value={comparison.store} />
        <MetadataField label="Created" value={formatDateTime(comparison.created_at)} />
        <MetadataField label="Git SHA" value={shortHash(comparison.git_sha)} />
        <MetadataField label="Strategies" value={comparison.per_strategy_stats.length} />
      </CardContent>
    </Card>
  );
}

function specsFromComparison(comparison: ComparisonDetail): OverlaySpec[] {
  return comparison.per_strategy_stats.map((row) => ({
    label: row.strategy,
    experimentId: row.experiment_id,
  }));
}

export function ComparisonDetailPage() {
  const { name = "" } = useParams<{ name: string }>();
  const query = useComparison(name);

  return (
    <QueryRenderer
      query={query}
      errorTitle="Failed to load comparison"
      loadingMessage="Loading comparison..."
    >
      {(comparison) => (
        <div className="flex flex-col gap-4">
          <BackLink to={ROUTES.comparisons}>All comparisons</BackLink>
          <IdentityCard comparison={comparison} />

          <Card>
            <CardHeader>
              <CardTitle>Per-strategy stats</CardTitle>
            </CardHeader>
            <CardContent>
              <PerStrategyStatsTable rows={comparison.per_strategy_stats} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Equity overlay (chained walk-forward)</CardTitle>
            </CardHeader>
            <CardContent>
              <EquityOverlayChart specs={specsFromComparison(comparison)} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Static figures</CardTitle>
            </CardHeader>
            <CardContent>
              <PlotIndex
                plots={comparison.plots}
                urlForPlot={(plotName) => comparisonPlotDownloadUrl(comparison.name, plotName)}
                emptyMessage="No static figures produced for this comparison."
              />
            </CardContent>
          </Card>
        </div>
      )}
    </QueryRenderer>
  );
}
