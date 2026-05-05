import { useParams } from "react-router-dom";
import { regimePlotDownloadUrl, useRegimeReport, type RegimeReportDetail } from "@/api/regime";
import { BackLink } from "@/components/BackLink";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RegimeMetricHeatmap } from "@/components/charts/RegimeMetricHeatmap";
import { RegimeTimeline } from "@/components/charts/RegimeTimeline";
import { MetadataField } from "@/components/MetadataField";
import { PerRegimeStatsTable } from "@/components/regime/PerRegimeStatsTable";
import { PlotIndex } from "@/components/PlotIndex";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, shortHash } from "@/lib/format";
import { runDetailPath, ROUTES } from "@/lib/routes";
import { Link } from "react-router-dom";

function IdentityCard({ regime }: { regime: RegimeReportDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{regime.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetadataField label="Store" value={regime.store} />
        <MetadataField label="Created" value={formatDateTime(regime.created_at)} />
        <MetadataField label="Git SHA" value={shortHash(regime.git_sha)} />
        <MetadataField label="Detector kind" value={regime.kind} />
        <MetadataField label="Detector name" value={regime.detector_name} />
        <MetadataField
          label="Source run"
          value={
            <Link
              to={runDetailPath(regime.experiment_id)}
              className="text-primary hover:underline font-mono"
            >
              {regime.experiment_id}
            </Link>
          }
        />
      </CardContent>
    </Card>
  );
}

export function RegimeDetailPage() {
  const { name = "" } = useParams<{ name: string }>();
  const query = useRegimeReport(name);

  return (
    <QueryRenderer
      query={query}
      errorTitle="Failed to load regime report"
      loadingMessage="Loading regime report…"
    >
      {(regime) => (
        <div className="flex flex-col gap-4">
          <BackLink to={ROUTES.regime}>All regime reports</BackLink>
          <IdentityCard regime={regime} />

          <Card>
            <CardHeader>
              <CardTitle>Regime timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <RegimeTimeline slices={regime.slices} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Per-regime stats</CardTitle>
            </CardHeader>
            <CardContent>
              <PerRegimeStatsTable rows={regime.per_regime_stats} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Metric heatmap</CardTitle>
            </CardHeader>
            <CardContent>
              <RegimeMetricHeatmap rows={regime.per_regime_stats} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Static figures</CardTitle>
            </CardHeader>
            <CardContent>
              <PlotIndex
                plots={regime.plots}
                urlForPlot={(plotName) => regimePlotDownloadUrl(regime.name, plotName)}
                emptyMessage="No static figures produced for this regime report."
              />
            </CardContent>
          </Card>
        </div>
      )}
    </QueryRenderer>
  );
}
