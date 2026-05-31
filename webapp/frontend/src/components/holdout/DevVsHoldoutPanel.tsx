import { Link } from "react-router-dom";
import type { HoldoutEvalDetail } from "@/api/holdout";
import { MetadataField } from "@/components/MetadataField";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatMetric, formatPercent } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";
import { isRunSource, sourceKindLabel } from "@/lib/sourceKind";

function SourceLabel({ holdout }: { holdout: HoldoutEvalDetail }) {
  if (isRunSource(holdout.source_kind)) {
    return (
      <Link to={runDetailPath(holdout.source_id)} className="text-primary hover:underline">
        {holdout.source_id}
      </Link>
    );
  }
  return <span>{holdout.source_id}</span>;
}

export function DevVsHoldoutPanel({ holdout }: { holdout: HoldoutEvalDetail }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4" data-testid="dev-vs-holdout-panel">
      <Card>
        <CardHeader>
          <CardTitle>Dev source</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <MetadataField label="Source kind" value={sourceKindLabel(holdout.source_kind)} />
          <MetadataField label="Source ID" value={<SourceLabel holdout={holdout} />} />
          <MetadataField label="Source path" value={holdout.source_path} />
          <MetadataField label="Dev bars" value={holdout.n_dev_bars} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Holdout metrics</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <MetadataField label="Holdout bars" value={holdout.n_holdout_bars} />
          <MetadataField label="Total return" value={formatPercent(holdout.total_return)} />
          <MetadataField
            label="Annualized return"
            value={formatPercent(holdout.annualized_return)}
          />
          <MetadataField label="Sharpe" value={formatMetric(holdout.sharpe_ratio)} />
          <MetadataField label="Sortino" value={formatMetric(holdout.sortino_ratio)} />
          <MetadataField label="Calmar" value={formatMetric(holdout.calmar_ratio)} />
          <MetadataField label="Max DD" value={formatPercent(holdout.max_drawdown)} />
          <MetadataField label="Win rate" value={formatPercent(holdout.win_rate)} />
          <MetadataField label="Trades" value={holdout.trade_count} />
        </CardContent>
      </Card>
    </div>
  );
}
