import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { Pencil } from "lucide-react";
import {
  useDeployment,
  useDeploymentSignals,
  usePredictIfStale,
  useRenameDeployment,
  useSignalEvaluation,
  type CostScenario,
  type DeploymentDetail,
  type ScoredSignalOut,
  type SignalEvaluationOut,
  type SignalRowOut,
} from "@/api/deployments";
import { BackLink } from "@/components/BackLink";
import { EquityChart } from "@/components/charts/EquityChart";
import { QueryRenderer } from "@/components/QueryRenderer";
import { SignalBadge } from "@/components/SignalBadge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { formatDate, formatMetric, formatPercent } from "@/lib/format";
import { ROUTES } from "@/lib/routes";
import { signalKindForStrategy, type SignalKind } from "@/lib/signalKind";
import { sourceKindLabel } from "@/lib/sourceKind";

export function DeploymentDetailPage() {
  const { deploymentId = "" } = useParams<{ deploymentId: string }>();
  const query = useDeployment(deploymentId);

  return (
    <div className="flex flex-col gap-4">
      <BackLink to={ROUTES.deployments}>All deployments</BackLink>
      <QueryRenderer
        query={query}
        errorTitle="Failed to load deployment"
        loadingMessage="Loading deployment..."
      >
        {(deployment) => <DeploymentBody deployment={deployment} />}
      </QueryRenderer>
    </div>
  );
}

function DeploymentBody({ deployment }: { deployment: DeploymentDetail }) {
  const [costTier, setCostTier] = useState<CostScenario>("normal");
  const predict = usePredictIfStale(deployment.id);
  const signalsQuery = useDeploymentSignals(deployment.id);
  const evaluationQuery = useSignalEvaluation(deployment.id, costTier);
  const { mutate: runPredict } = predict;

  useEffect(() => {
    runPredict();
  }, [deployment.id, runPredict]);

  const latest: SignalRowOut | null = predict.data?.signal ?? deployment.latest_signal ?? null;
  const computing = predict.isPending && latest === null;
  const kind = signalKindForStrategy(deployment.strategy_name);

  return (
    <>
      <DeploymentHeader deployment={deployment} />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Today&apos;s signal</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-4">
            <SignalBadge signal={latest?.signal ?? null} loading={computing} kind={kind} />
            {latest !== null && (
              <span className="text-xs text-muted-foreground">
                Signal date {formatDate(latest.signal_date)}
              </span>
            )}
          </div>
          {kind === "leverage" && (
            <p className="text-xs text-muted-foreground">
              Position-size multiplier (x notional exposure), not a fixed long/short - e.g. 1.39x =
              139% long, 0x = flat.
            </p>
          )}
          {predict.isError && (
            <Alert variant="destructive">
              <AlertTitle>Could not compute a signal</AlertTitle>
              <AlertDescription>{predict.error.message}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
          <CardTitle className="text-lg">Signal performance</CardTitle>
          <CostTierSelector value={costTier} onChange={setCostTier} />
        </CardHeader>
        <CardContent>
          <QueryRenderer
            query={evaluationQuery}
            errorTitle="Failed to load signal evaluation"
            loadingMessage="Scoring signals..."
          >
            {(evaluation) => <SignalPerformanceBody evaluation={evaluation} />}
          </QueryRenderer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Signal history</CardTitle>
        </CardHeader>
        <CardContent>
          <QueryRenderer query={signalsQuery} errorTitle="Failed to load signal history">
            {(signals) => (
              <SignalHistoryTable
                signals={signals}
                kind={kind}
                evaluation={evaluationQuery.data ?? null}
              />
            )}
          </QueryRenderer>
        </CardContent>
      </Card>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-base">{value}</span>
    </div>
  );
}

function pctOrDash(value: number | null): string {
  return value === null ? "-" : formatPercent(value);
}

const COST_TIERS: readonly { value: CostScenario; label: string }[] = [
  { value: "zero", label: "Zero" },
  { value: "low", label: "Low" },
  { value: "normal", label: "Normal" },
  { value: "high", label: "High" },
];

function CostTierSelector({
  value,
  onChange,
}: {
  value: CostScenario;
  onChange: (next: CostScenario) => void;
}) {
  return (
    <div className="flex items-center gap-1" data-testid="cost-tier-selector">
      <span className="mr-1 text-xs text-muted-foreground">Costs</span>
      {COST_TIERS.map((tier) => (
        <Button
          key={tier.value}
          size="sm"
          variant={tier.value === value ? "default" : "ghost"}
          onClick={() => {
            onChange(tier.value);
          }}
        >
          {tier.label}
        </Button>
      ))}
    </div>
  );
}

function SignalPerformanceBody({ evaluation }: { evaluation: SignalEvaluationOut }) {
  const { grossEquity, netEquity } = useMemo(() => {
    const scored = evaluation.rows.filter((row) => row.scored);
    return {
      grossEquity: [1, ...scored.map((row) => 1 + (row.cumulative_return ?? 0))],
      netEquity: [1, ...scored.map((row) => 1 + (row.net_cumulative_return ?? 0))],
    };
  }, [evaluation.rows]);

  if (evaluation.n_scored === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No signals scored yet - a signal is scored open-to-open once its next session opens.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4" data-testid="signal-performance">
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        <Stat label="Hit rate" value={pctOrDash(evaluation.hit_rate)} />
        <Stat label="Cumulative (gross)" value={pctOrDash(evaluation.cumulative_return)} />
        <Stat label="Cumulative (net)" value={pctOrDash(evaluation.net_cumulative_return)} />
        <Stat
          label="Scored"
          value={`${String(evaluation.n_scored)} / ${String(evaluation.n_signals)}`}
        />
      </div>
      <EquityChart
        traces={[
          { name: "Gross", equity: grossEquity },
          { name: "Net of costs", equity: netEquity },
        ]}
        height={260}
        xLabel="Scored signal"
        yLabel="Cumulative (x)"
      />
    </div>
  );
}

function DeploymentHeader({ deployment }: { deployment: DeploymentDetail }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(deployment.name);
  const rename = useRenameDeployment(deployment.id);

  function startEdit() {
    setDraft(deployment.name);
    rename.reset();
    setEditing(true);
  }

  function save() {
    const next = draft.trim();
    if (next.length === 0 || next === deployment.name) {
      setEditing(false);
      return;
    }
    rename.mutate(next, {
      onSuccess: () => {
        setEditing(false);
      },
    });
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-3">
        {editing ? (
          <div className="flex flex-1 items-center gap-2">
            <Input
              autoFocus
              value={draft}
              maxLength={200}
              onChange={(e) => {
                setDraft(e.target.value);
              }}
              data-testid="deployment-rename-input"
              className="max-w-md"
            />
            <Button size="sm" onClick={save} disabled={rename.isPending}>
              Save
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setEditing(false);
              }}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <>
            <h2 className="text-2xl font-semibold tracking-tight" data-testid="deployment-name">
              {deployment.name}
            </h2>
            <Button
              size="sm"
              variant="ghost"
              onClick={startEdit}
              data-testid="deployment-rename-cta"
            >
              <Pencil className="mr-1 h-3 w-3" />
              Rename
            </Button>
          </>
        )}
      </div>
      {rename.isError && <p className="text-sm text-destructive">{rename.error.message}</p>}
      <p className="text-sm text-muted-foreground">
        {deployment.ticker} | {deployment.strategy_name} | {deployment.interval} | trained through{" "}
        {formatDate(deployment.train_end)}
      </p>
      <p className="text-xs text-muted-foreground">
        Source: {sourceKindLabel(deployment.source_kind)} | {deployment.source_id}
      </p>
    </div>
  );
}

function ReturnCell({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className="text-muted-foreground">-</span>;
  }
  const tone =
    value > 0
      ? "text-emerald-600 dark:text-emerald-400"
      : value < 0
        ? "text-rose-600 dark:text-rose-400"
        : "text-muted-foreground";
  return <span className={tone}>{formatPercent(value)}</span>;
}

// A signal moves through three lifecycle states as sessions open:
//   pending  - its entry session hasn't opened, nothing to act on yet.
//   holding  - entered at the open; position is live, exit (hence score) pending.
//   scored   - exit session opened, realised return + hit are known.
function StatusCell({ score }: { score: ScoredSignalOut | undefined }) {
  if (score?.scored) {
    if (score.hit === null) {
      return <span className="text-muted-foreground">flat</span>;
    }
    return score.hit ? (
      <span className="text-emerald-600 dark:text-emerald-400">win</span>
    ) : (
      <span className="text-rose-600 dark:text-rose-400">loss</span>
    );
  }
  if (score?.entry_open != null) {
    return (
      <span
        className="text-amber-600 dark:text-amber-400"
        title="Entered at the open - held now; score lands at the next session's open."
      >
        holding
      </span>
    );
  }
  return (
    <span
      className="text-muted-foreground"
      title="Not entered yet - enters at this signal's date open."
    >
      pending
    </span>
  );
}

function SignalHistoryTable({
  signals,
  kind,
  evaluation,
}: {
  signals: readonly SignalRowOut[];
  kind: SignalKind;
  evaluation: SignalEvaluationOut | null;
}) {
  const scoreByDate = useMemo(() => {
    const map = new Map<string, ScoredSignalOut>();
    for (const row of evaluation?.rows ?? []) {
      map.set(formatDate(row.bar_ts), row);
    }
    return map;
  }, [evaluation]);
  const newestFirst = useMemo(() => [...signals].reverse(), [signals]);

  if (signals.length === 0) {
    return <p className="text-sm text-muted-foreground">No signals recorded yet.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="signal-history-table">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4">Signal date</th>
            <th className="py-2 pr-4">Signal</th>
            <th className="py-2 pr-4 text-right">Entry open</th>
            <th className="py-2 pr-4 text-right">Asset move</th>
            <th className="py-2 pr-4 text-right">Gross return</th>
            <th className="py-2 pr-4 text-right">Net return</th>
            <th className="py-2 pl-8 pr-4">Status</th>
            <th className="py-2 text-right">Warmup bars</th>
          </tr>
        </thead>
        <tbody>
          {newestFirst.map((row) => {
            const score = scoreByDate.get(formatDate(row.bar_ts));
            const scored = score?.scored ?? false;
            return (
              <tr key={`${row.bar_ts}-${row.submitted_at}`} className="border-b last:border-0">
                <td className="py-2 pr-4 font-mono text-xs">{formatDate(row.signal_date)}</td>
                <td className="py-2 pr-4">
                  <SignalBadge signal={row.signal} kind={kind} />
                </td>
                <td className="py-2 pr-4 text-right font-mono">
                  {formatMetric(score?.entry_open, 2)}
                </td>
                <td className="py-2 pr-4 text-right font-mono">
                  <ReturnCell value={scored ? score?.asset_return : null} />
                </td>
                <td className="py-2 pr-4 text-right font-mono">
                  <ReturnCell value={scored ? score?.listened_return : null} />
                </td>
                <td className="py-2 pr-4 text-right font-mono">
                  <ReturnCell value={scored ? score?.net_listened_return : null} />
                </td>
                <td className="py-2 pl-8 pr-4">
                  <StatusCell score={score} />
                </td>
                <td className="py-2 text-right font-mono">{row.warmup_bars_used}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
