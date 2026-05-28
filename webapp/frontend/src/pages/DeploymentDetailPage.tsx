import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Pencil } from "lucide-react";
import {
  useDeployment,
  useDeploymentSignals,
  usePredictIfStale,
  useRenameDeployment,
  type DeploymentDetail,
  type SignalRowOut,
} from "@/api/deployments";
import { BackLink } from "@/components/BackLink";
import { QueryRenderer } from "@/components/QueryRenderer";
import { SignalBadge } from "@/components/SignalBadge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { formatDate } from "@/lib/format";
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
        loadingMessage="Loading deployment…"
      >
        {(deployment) => <DeploymentBody deployment={deployment} />}
      </QueryRenderer>
    </div>
  );
}

function DeploymentBody({ deployment }: { deployment: DeploymentDetail }) {
  const predict = usePredictIfStale(deployment.id);
  const signalsQuery = useDeploymentSignals(deployment.id);
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
                Signal date {formatDate(latest.bar_ts)}
              </span>
            )}
          </div>
          {kind === "leverage" && (
            <p className="text-xs text-muted-foreground">
              Position-size multiplier (× notional exposure), not a fixed long/short — e.g. 1.39× =
              139% long, 0× = flat.
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
        <CardHeader>
          <CardTitle className="text-lg">Signal history</CardTitle>
        </CardHeader>
        <CardContent>
          <QueryRenderer query={signalsQuery} errorTitle="Failed to load signal history">
            {(signals) => <SignalHistoryTable signals={signals} kind={kind} />}
          </QueryRenderer>
        </CardContent>
      </Card>
    </>
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
        {deployment.ticker} · {deployment.strategy_name} · {deployment.interval} · trained through{" "}
        {formatDate(deployment.train_end)}
      </p>
      <p className="text-xs text-muted-foreground">
        Source: {sourceKindLabel(deployment.source_kind)} · {deployment.source_id}
      </p>
    </div>
  );
}

function SignalHistoryTable({
  signals,
  kind,
}: {
  signals: readonly SignalRowOut[];
  kind: SignalKind;
}) {
  if (signals.length === 0) {
    return <p className="text-sm text-muted-foreground">No signals recorded yet.</p>;
  }
  const newestFirst = [...signals].reverse();

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="signal-history-table">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4">Signal date</th>
            <th className="py-2 pr-4">Signal</th>
            <th className="py-2 text-right">Warmup bars</th>
          </tr>
        </thead>
        <tbody>
          {newestFirst.map((row) => (
            <tr key={`${row.bar_ts}-${row.submitted_at}`} className="border-b last:border-0">
              <td className="py-2 pr-4 font-mono text-xs">{formatDate(row.bar_ts)}</td>
              <td className="py-2 pr-4">
                <SignalBadge signal={row.signal} kind={kind} />
              </td>
              <td className="py-2 text-right font-mono">{row.warmup_bars_used}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
