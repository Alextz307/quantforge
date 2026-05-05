import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RunDetail } from "@/api/runs";
import { MetadataField } from "@/components/MetadataField";
import { formatDateTime, shortHash } from "@/lib/format";

export function ManifestPanel({ run }: { run: RunDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{run.name}</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetadataField label="Experiment ID" value={run.experiment_id} />
        <MetadataField label="Strategy" value={run.strategy} />
        <MetadataField label="Store" value={run.store} />
        <MetadataField label="Tickers" value={run.tickers.join(", ")} />
        <MetadataField label="Interval" value={run.interval} />
        <MetadataField label="Created" value={formatDateTime(run.created_at)} />
        <MetadataField label="Git SHA" value={shortHash(run.git_sha)} />
        <MetadataField label="Seed" value={run.seed} />
        <MetadataField label="Data hash" value={shortHash(run.data_hash)} />
        <MetadataField label="Slippage" value={run.slippage_scenario} />
        <MetadataField
          label="Holdout start"
          value={run.holdout_start ? formatDateTime(run.holdout_start) : "—"}
        />
        <MetadataField label="Pretrained leaves" value={run.pretrained_leaves.length} />
      </CardContent>
    </Card>
  );
}
