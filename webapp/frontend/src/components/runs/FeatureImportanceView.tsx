import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useJob, useSubmitJob, type JobStatus } from "@/api/jobs";
import { queryKeys } from "@/api/queryKeys";
import type { FeatureImportanceResponse } from "@/api/runs";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { JobStatusPill } from "@/components/jobs/JobStatusPill";
import { useJobStream } from "@/hooks/useJobStream";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { runDetailPath } from "@/lib/routes";

const COMPUTE_EXPLAINER =
  "Computing importance re-trains this strategy on the run's exact config (same seed, cached " +
  "data). If the re-fit reproduces this run's metrics, importance is attached here. If training " +
  "is non-deterministic and the re-fit diverges, it's saved as a separate run instead, so this " +
  "run's reported metrics stay consistent with its own models.";

// The in-flight job id lives in sessionStorage (not the URL) so a reload mid-
// computation resumes the watcher, while a compute click avoids a URL change -
// which would re-render the route subtree and flicker the Plotly charts.
const IMPORTANCE_JOB_STORAGE_PREFIX = "importanceJob:";

function importanceJobKey(experimentId: string): string {
  return `${IMPORTANCE_JOB_STORAGE_PREFIX}${experimentId}`;
}

export interface FeatureImportanceViewProps {
  experimentId: string;
  strategyName: string;
  response: FeatureImportanceResponse;
}

/**
 * Decides what the feature-importance card shows: the chart when importance is
 * present, a "compute" action when the run's strategy supports importance but
 * none was computed, or an explanation for rule-based strategies that have none.
 */
export function FeatureImportanceView({
  experimentId,
  strategyName,
  response,
}: FeatureImportanceViewProps) {
  if (response.entries.length > 0) {
    return <FeatureImportanceChart response={response} />;
  }
  if (!response.computable) {
    return (
      <div
        data-testid="feature-importance-not-applicable"
        className="mx-auto max-w-prose py-10 text-center text-sm text-muted-foreground"
      >
        Feature importance doesn&apos;t apply to {strategyName}. It&apos;s a rule-based strategy
        that generates signals from explicit trading rules, not a model trained on engineered
        features, so there are no feature columns to rank. The four model-based strategies
        (ReturnForecast, VolatilityTargeting, MomentumGatekeeper, CrossAssetMomentum) are the ones
        with importance to compute.
      </div>
    );
  }
  return (
    <ComputeImportance
      experimentId={experimentId}
      divergedRunId={response.diverged_run_id ?? null}
    />
  );
}

function DivergedNotice({ runId, fromExperimentId }: { runId: string; fromExperimentId: string }) {
  return (
    <Alert data-testid="feature-importance-diverged">
      <AlertTitle>Importance saved as a separate run</AlertTitle>
      <AlertDescription>
        A recompute diverged from this run's metrics (training is non-deterministic on this device),
        so importance was computed on a fresh re-fit and saved as a separate run rather than
        attached here, leaving this run's metrics consistent with its own models.{" "}
        <Link
          className="underline"
          to={runDetailPath(runId)}
          state={{ from: runDetailPath(fromExperimentId) }}
          data-testid="feature-importance-diverged-link"
        >
          View the importance run &rarr;
        </Link>
      </AlertDescription>
    </Alert>
  );
}

function ComputeImportance({
  experimentId,
  divergedRunId,
}: {
  experimentId: string;
  divergedRunId: string | null;
}) {
  const submit = useSubmitJob();
  const [liveJobId, setLiveJobId] = useState<string | null>(() =>
    sessionStorage.getItem(importanceJobKey(experimentId)),
  );

  const startWatching = useCallback(
    (jobId: string) => {
      sessionStorage.setItem(importanceJobKey(experimentId), jobId);
      setLiveJobId(jobId);
    },
    [experimentId],
  );

  const reset = useCallback(() => {
    sessionStorage.removeItem(importanceJobKey(experimentId));
    setLiveJobId(null);
  }, [experimentId]);

  if (liveJobId !== null) {
    return <ImportanceJobWatcher jobId={liveJobId} experimentId={experimentId} onReset={reset} />;
  }

  return (
    <div data-testid="feature-importance-compute" className="space-y-3 py-6">
      {divergedRunId !== null ? (
        <DivergedNotice runId={divergedRunId} fromExperimentId={experimentId} />
      ) : (
        <p className="text-sm text-muted-foreground">{COMPUTE_EXPLAINER}</p>
      )}
      <Button
        type="button"
        size="sm"
        variant={divergedRunId !== null ? "outline" : "default"}
        disabled={submit.isPending}
        data-testid="feature-importance-compute-button"
        onClick={() => {
          submit.mutate(
            {
              kind: "importance",
              importance_payload: { run_id: experimentId },
              feature_importance: false,
            },
            {
              onSuccess: (row) => {
                startWatching(row.id);
              },
            },
          );
        }}
      >
        {submit.isPending
          ? "Starting…"
          : divergedRunId !== null
            ? "Recompute importance"
            : "Compute importance"}
      </Button>
      {submit.isError && (
        <Alert variant="destructive">
          <AlertDescription data-testid="feature-importance-compute-error">
            {submit.error.message}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

function ImportanceJobWatcher({
  jobId,
  experimentId,
  onReset,
}: {
  jobId: string;
  experimentId: string;
  onReset: () => void;
}) {
  const qc = useQueryClient();
  // Resume an in-flight job (e.g. after a reload) from its stored id; "running"
  // is only the pre-fetch placeholder until useJob + the stream report the truth.
  useJobStream(jobId, "running");
  const jobQuery = useJob(jobId);
  const job = jobQuery.data;
  const status: JobStatus = job?.status ?? "running";
  const settled =
    jobQuery.isError || status === "completed" || status === "failed" || status === "cancelled";

  useEffect(() => {
    // Stop persisting a finished/gone job so a later reload doesn't resume it;
    // the live render holds the outcome until the parent's refetch supersedes it.
    if (settled) sessionStorage.removeItem(importanceJobKey(experimentId));
  }, [settled, experimentId]);

  useEffect(() => {
    // Stored id points at a job that no longer exists; reset so the persistent
    // state (chart / pointer / button) takes over.
    if (jobQuery.isError) onReset();
  }, [jobQuery.isError, onReset]);

  useEffect(() => {
    if (status !== "completed") return;
    // Hand back to ComputeImportance only once the refetch lands - resetting
    // earlier would re-render the parent with the stale pre-recompute response,
    // flashing the "Compute importance" button before the final result.
    let active = true;
    void qc
      .invalidateQueries({ queryKey: queryKeys.runFeatureImportance(experimentId) })
      .finally(() => {
        if (active) onReset();
      });
    return () => {
      active = false;
    };
  }, [status, qc, experimentId, onReset]);

  if (status === "failed" || status === "cancelled") {
    return (
      <Alert variant="destructive" data-testid="feature-importance-job-failed">
        <AlertTitle>Importance computation {status}</AlertTitle>
        <AlertDescription className="space-y-3">
          <p>The re-run did not finish. Open the job under Jobs for the full log.</p>
          <Button
            type="button"
            size="sm"
            variant="outline"
            data-testid="feature-importance-retry"
            onClick={onReset}
          >
            Try again
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (status === "completed") {
    const divergedRunId =
      job != null && job.experiment_id != null && job.experiment_id !== experimentId
        ? job.experiment_id
        : null;
    if (divergedRunId !== null) {
      return <DivergedNotice runId={divergedRunId} fromExperimentId={experimentId} />;
    }
    return (
      <div
        data-testid="feature-importance-backfilled"
        className="text-sm text-muted-foreground py-6 text-center"
      >
        Importance computed. Loading&hellip;
      </div>
    );
  }

  return (
    <div data-testid="feature-importance-running" className="flex items-center gap-3 py-6">
      <JobStatusPill status={status} />
      <span className="text-sm text-muted-foreground">
        Re-training the strategy to compute importance&hellip;
      </span>
    </div>
  );
}
