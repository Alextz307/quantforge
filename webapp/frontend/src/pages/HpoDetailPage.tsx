import { Link, useParams } from "react-router-dom";
import { useHpoParamImportance, useHpoStudy, useHpoTrials, type HpoDetail } from "@/api/hpo";
import { BackLink } from "@/components/BackLink";
import { ConnectionIndicator } from "@/components/ConnectionIndicator";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HpoConvergenceChart } from "@/components/charts/HpoConvergenceChart";
import { HpoParamImportanceChart } from "@/components/charts/HpoParamImportanceChart";
import { BestConfigCard } from "@/components/hpo/BestConfigCard";
import { TrialTable } from "@/components/hpo/TrialTable";
import { MetadataField } from "@/components/MetadataField";
import { QueryRenderer } from "@/components/QueryRenderer";
import { useHpoTrialStream } from "@/hooks/useHpoTrialStream";
import { formatDateTime, formatMetric } from "@/lib/format";
import { ROUTES } from "@/lib/routes";
import { SOURCE_KIND_HPO } from "@/lib/sourceKind";

function IdentityCard({ study }: { study: HpoDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{study.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetadataField label="Store" value={study.store} />
        <MetadataField label="Created" value={formatDateTime(study.created_at)} />
        <MetadataField label="Direction" value={study.direction} />
        <MetadataField label="Total trials" value={study.n_trials} />
        <MetadataField label="Completed" value={study.n_complete} />
        <MetadataField label="Best value" value={formatMetric(study.best_value)} />
        <MetadataField label="Best trial #" value={study.best_trial_number ?? "-"} />
      </CardContent>
    </Card>
  );
}

export function HpoDetailPage() {
  const { wireId = "" } = useParams<{ wireId: string }>();
  const studyQuery = useHpoStudy(wireId, { livePoll: true });
  const isLive = studyQuery.data?.live_job_id != null;
  const trialsQuery = useHpoTrials(wireId);
  const importanceQuery = useHpoParamImportance(wireId, { isLive });
  const stream = useHpoTrialStream(wireId);

  return (
    <QueryRenderer
      query={studyQuery}
      errorTitle="Failed to load HPO study"
      loadingMessage="Loading HPO study..."
    >
      {(study) => (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <BackLink to={ROUTES.hpo}>All HPO studies</BackLink>
            {study.best_config_reserves_holdout && (
              <Button asChild variant="outline" size="sm">
                <Link
                  to={`${ROUTES.configureHoldout}?source_kind=${SOURCE_KIND_HPO}&source_id=${encodeURIComponent(study.wire_id)}`}
                  data-testid="hpo-detail-holdout-cta"
                >
                  Run holdout eval
                </Link>
              </Button>
            )}
          </div>
          {isLive && <ConnectionIndicator state={stream.connection} className="self-start" />}
          <IdentityCard study={study} />

          <Card>
            <CardHeader>
              <CardTitle>Convergence</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryRenderer
                query={trialsQuery}
                errorTitle="Failed to load trials"
                loadingMessage="Loading trials..."
              >
                {(trials) => <HpoConvergenceChart trials={trials} direction={study.direction} />}
              </QueryRenderer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Hyperparameter importance</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryRenderer
                query={importanceQuery}
                errorTitle="Failed to load importance"
                loadingMessage="Loading importance..."
              >
                {(response) => <HpoParamImportanceChart response={response} />}
              </QueryRenderer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Trials</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryRenderer
                query={trialsQuery}
                errorTitle="Failed to load trials"
                loadingMessage="Loading trials..."
              >
                {(trials) => (
                  <TrialTable trials={trials} bestTrialNumber={study.best_trial_number} />
                )}
              </QueryRenderer>
            </CardContent>
          </Card>

          <BestConfigCard config={study.best_config} />
        </div>
      )}
    </QueryRenderer>
  );
}
