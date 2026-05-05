import { Link } from "react-router-dom";
import type { PerStrategyStatsRow } from "@/api/comparisons";
import { formatPercent, withCi } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";

export function PerStrategyStatsTable({ rows }: { rows: readonly PerStrategyStatsRow[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No per-strategy stats recorded.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="per-strategy-stats-table">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4">Strategy</th>
            <th className="py-2 pr-4 text-right">Folds</th>
            <th className="py-2 pr-4 text-right">Sharpe (mean [95% CI])</th>
            <th className="py-2 pr-4 text-right">Sortino (mean [95% CI])</th>
            <th className="py-2 pr-4 text-right">Calmar (mean [95% CI])</th>
            <th className="py-2 pr-4 text-right">Total return</th>
            <th className="py-2 pr-4 text-right">Max DD (worst)</th>
            <th className="py-2 pr-4 text-right">Win rate</th>
            <th className="py-2 pr-0 text-right">Trades</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.strategy} className="border-b last:border-0">
              <td className="py-2 pr-4 font-mono">
                <Link
                  to={runDetailPath(row.experiment_id)}
                  className="text-primary hover:underline"
                >
                  {row.strategy}
                </Link>
              </td>
              <td className="py-2 pr-4 text-right font-mono">{row.n_folds}</td>
              <td className="py-2 pr-4 text-right font-mono">
                {withCi(row.sharpe_mean, row.sharpe_ci95_low, row.sharpe_ci95_high)}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {withCi(row.sortino_mean, row.sortino_ci95_low, row.sortino_ci95_high)}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {withCi(row.calmar_mean, row.calmar_ci95_low, row.calmar_ci95_high)}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {formatPercent(row.total_return_mean)}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {formatPercent(row.max_drawdown_worst)}
              </td>
              <td className="py-2 pr-4 text-right font-mono">{formatPercent(row.win_rate_mean)}</td>
              <td className="py-2 pr-0 text-right font-mono">{row.trade_count_total}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
