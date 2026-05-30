import type { FoldRow } from "@/api/runs";
import { formatDate, formatMetric, formatPercent } from "@/lib/format";

export function FoldMetricsTable({ folds }: { folds: readonly FoldRow[] }) {
  if (folds.length === 0) {
    return <p className="text-sm text-muted-foreground">No fold metrics available.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="fold-metrics-table">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4">Fold</th>
            <th className="py-2 pr-4">Train</th>
            <th className="py-2 pr-4">Test</th>
            <th className="py-2 pr-4 text-right">Total return</th>
            <th className="py-2 pr-4 text-right">Sharpe</th>
            <th className="py-2 pr-4 text-right">Sortino</th>
            <th className="py-2 pr-4 text-right">Calmar</th>
            <th className="py-2 pr-4 text-right">Max DD</th>
            <th className="py-2 pr-4 text-right">Win rate</th>
            <th className="py-2 pr-0 text-right">Trades</th>
          </tr>
        </thead>
        <tbody>
          {folds.map((f) => (
            <tr key={f.fold_index} className="border-b last:border-0">
              <td className="py-2 pr-4 font-mono">{f.fold_index}</td>
              <td className="py-2 pr-4 font-mono text-xs">
                {formatDate(f.train_start)} to {formatDate(f.train_end)}
              </td>
              <td className="py-2 pr-4 font-mono text-xs">
                {formatDate(f.test_start)} to {formatDate(f.test_end)}
              </td>
              <td className="py-2 pr-4 text-right font-mono">{formatPercent(f.total_return)}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatMetric(f.sharpe_ratio)}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatMetric(f.sortino_ratio)}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatMetric(f.calmar_ratio)}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatPercent(f.max_drawdown)}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatPercent(f.win_rate)}</td>
              <td className="py-2 pr-0 text-right font-mono">{f.trade_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
