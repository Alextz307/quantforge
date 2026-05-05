import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RunDetail } from "@/api/runs";
import { formatDateTime, shortHash } from "@/lib/format";

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="text-sm font-mono break-all">{value}</span>
    </div>
  );
}

export function ManifestPanel({ run }: { run: RunDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{run.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <Field label="Experiment ID" value={run.experiment_id} />
        <Field label="Strategy" value={run.strategy} />
        <Field label="Store" value={run.store} />
        <Field label="Tickers" value={run.tickers.join(", ")} />
        <Field label="Interval" value={run.interval} />
        <Field label="Created" value={formatDateTime(run.created_at)} />
        <Field label="Git SHA" value={shortHash(run.git_sha)} />
        <Field label="Seed" value={run.seed} />
        <Field label="Data hash" value={shortHash(run.data_hash)} />
        <Field label="Slippage" value={run.slippage_scenario} />
        <Field
          label="Holdout start"
          value={run.holdout_start ? formatDateTime(run.holdout_start) : "—"}
        />
        <Field label="Pretrained leaves" value={run.pretrained_leaves.length} />
      </CardContent>
    </Card>
  );
}
