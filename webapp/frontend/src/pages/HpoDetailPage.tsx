import { useParams } from "react-router-dom";
import { useHpoStudy, useHpoTrials, type HpoDetail } from "@/api/hpo";
import { BackLink } from "@/components/BackLink";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HpoConvergenceChart } from "@/components/charts/HpoConvergenceChart";
import { BestConfigCard } from "@/components/hpo/BestConfigCard";
import { TrialTable } from "@/components/hpo/TrialTable";
import { MetadataField } from "@/components/MetadataField";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, formatMetric } from "@/lib/format";
import { ROUTES } from "@/lib/routes";

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
        <MetadataField
          label="Best trial #"
          value={study.best_trial_number ?? "—"}
        />
      </CardContent>
    </Card>
  );
}

export function HpoDetailPage() {
  const { name = "" } = useParams<{ name: string }>();
  const studyQuery = useHpoStudy(name);
  const trialsQuery = useHpoTrials(name);

  return (
    <QueryRenderer
      query={studyQuery}
      errorTitle="Failed to load HPO study"
      loadingMessage="Loading HPO study…"
    >
      {(study) => (
        <div className="flex flex-col gap-4">
          <BackLink to={ROUTES.hpo}>All HPO studies</BackLink>
          <IdentityCard study={study} />

          <Card>
            <CardHeader>
              <CardTitle>Convergence</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryRenderer
                query={trialsQuery}
                errorTitle="Failed to load trials"
                loadingMessage="Loading trials…"
              >
                {(trials) => (
                  <HpoConvergenceChart trials={trials} direction={study.direction} />
                )}
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
                loadingMessage="Loading trials…"
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
