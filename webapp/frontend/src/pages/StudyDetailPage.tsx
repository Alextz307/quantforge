import { useParams } from "react-router-dom";
import {
  useGenerateStudyConsolidated,
  useStudy,
  useStudyConsolidated,
  type StudyDetail,
} from "@/api/studies";
import { BackLink } from "@/components/BackLink";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ConnectionIndicator } from "@/components/ConnectionIndicator";
import { ConsolidatedReportPanel } from "@/components/studies/ConsolidatedReportPanel";
import { LegStatusGrid } from "@/components/studies/LegStatusGrid";
import { MetadataField } from "@/components/MetadataField";
import { QueryRenderer } from "@/components/QueryRenderer";
import { useStudyStream } from "@/hooks/useStudyStream";
import { formatDateTime, formatPercent, shortHash } from "@/lib/format";
import { ROUTES } from "@/lib/routes";

function IdentityCard({ study }: { study: StudyDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{study.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetadataField label="Spec" value={study.spec_name} />
        <MetadataField label="Spec hash" value={shortHash(study.spec_hash)} />
        <MetadataField label="Started" value={formatDateTime(study.started_at)} />
        <MetadataField label="Total legs" value={study.total_legs} />
        <MetadataField label="Completed" value={study.completed_legs} />
        <MetadataField label="Completion" value={formatPercent(study.completion_pct / 100)} />
      </CardContent>
    </Card>
  );
}

function ConsolidatedSection({ name, hasReport }: { name: string; hasReport: boolean }) {
  // ``hasReport`` short-circuits the consolidated query: the backend already
  // told us via StudyDetail.has_consolidated_report whether manifest.json
  // exists, so we skip the GET entirely on the first-time-generate branch
  // instead of relying on a 404 to drive the UI.
  const query = useStudyConsolidated(name, { enabled: hasReport });
  const mutation = useGenerateStudyConsolidated(name);

  if (!hasReport) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Consolidated report</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <p className="text-sm text-muted-foreground">
            Not yet generated. This consolidates per-leg runs, holdout evaluations, and pairwise
            comparisons into cross-leg rankings, a strategy x universe heatmap, and a dev-vs-holdout
            scatter.
          </p>
          <div className="flex items-center gap-3">
            <Button
              type="button"
              size="sm"
              disabled={mutation.isPending}
              onClick={() => {
                mutation.mutate();
              }}
            >
              {mutation.isPending ? "Generating..." : "Generate report"}
            </Button>
            {mutation.isError ? (
              <span className="text-sm text-destructive">{mutation.error.message}</span>
            ) : null}
          </div>
        </CardContent>
      </Card>
    );
  }
  if (query.isPending) {
    return <p className="text-sm text-muted-foreground">Loading consolidated report...</p>;
  }
  if (query.isError) {
    return (
      <p className="text-sm text-destructive">
        Failed to load consolidated report: {query.error.message}
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <ConsolidatedReportPanel dto={query.data} studyDirName={name} />
      <div className="flex items-center gap-3 px-1">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={mutation.isPending}
          onClick={() => {
            mutation.mutate();
          }}
        >
          {mutation.isPending ? "Regenerating..." : "Regenerate report"}
        </Button>
        {mutation.isError ? (
          <span className="text-sm text-destructive">{mutation.error.message}</span>
        ) : null}
      </div>
    </div>
  );
}

export function StudyDetailPage() {
  const { name = "" } = useParams<{ name: string }>();
  const query = useStudy(name);
  const stream = useStudyStream(name);

  return (
    <QueryRenderer
      query={query}
      errorTitle="Failed to load study"
      loadingMessage="Loading study..."
    >
      {(study) => {
        const isLive = study.completed_legs < study.total_legs;
        return (
          <div className="flex flex-col gap-4">
            <BackLink to={ROUTES.studies}>All studies</BackLink>
            {isLive && <ConnectionIndicator state={stream.connection} className="self-start" />}
            <IdentityCard study={study} />

            <Card>
              <CardHeader>
                <CardTitle>Leg status</CardTitle>
              </CardHeader>
              <CardContent>
                <LegStatusGrid legs={study.legs} />
              </CardContent>
            </Card>

            <ConsolidatedSection name={study.name} hasReport={study.has_consolidated_report} />
          </div>
        );
      }}
    </QueryRenderer>
  );
}
