import { useMemo } from "react";
import { Link } from "react-router-dom";
import type { LegStateRow } from "@/api/studies";
import { runDetailPath } from "@/lib/routes";

function classifyLeg(leg: LegStateRow): "complete" | "errored" | "running" | "pending" {
  if (leg.error) return "errored";
  if (leg.is_complete) return "complete";
  if (leg.started_at) return "running";
  return "pending";
}

const STATUS_STYLES = {
  complete: "bg-green-100 text-green-900 dark:bg-green-900/40 dark:text-green-200",
  errored: "bg-red-100 text-red-900 dark:bg-red-900/40 dark:text-red-200",
  running: "bg-yellow-100 text-yellow-900 dark:bg-yellow-900/40 dark:text-yellow-200",
  pending: "bg-muted text-muted-foreground",
} as const;

function StatusPill({ leg }: { leg: LegStateRow }) {
  const status = classifyLeg(leg);
  const className = `inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[status]}`;
  if (status === "complete" && leg.run_experiment_id) {
    return (
      <Link
        to={runDetailPath(leg.run_experiment_id)}
        className={`${className} hover:underline`}
        data-testid={`leg-cell-${leg.leg_id}`}
      >
        complete
      </Link>
    );
  }
  return (
    <span className={className} data-testid={`leg-cell-${leg.leg_id}`}>
      {status}
    </span>
  );
}

interface PivotedRow {
  strategy: string;
  byUniverse: Map<string, LegStateRow>;
}

function pivot(legs: readonly LegStateRow[]): { universes: string[]; rows: PivotedRow[] } {
  const rowMap = new Map<string, PivotedRow>();
  const universeSet = new Set<string>();
  for (const leg of legs) {
    let row = rowMap.get(leg.strategy);
    if (!row) {
      row = { strategy: leg.strategy, byUniverse: new Map() };
      rowMap.set(leg.strategy, row);
    }
    row.byUniverse.set(leg.universe, leg);
    universeSet.add(leg.universe);
  }
  const rows = [...rowMap.values()].sort((a, b) => a.strategy.localeCompare(b.strategy));
  const universes = [...universeSet].sort();
  return { universes, rows };
}

export function LegStatusGrid({ legs }: { legs: readonly LegStateRow[] }) {
  const { universes, rows } = useMemo(() => pivot(legs), [legs]);

  if (legs.length === 0) {
    return <p className="text-sm text-muted-foreground">No legs in this study.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="leg-status-grid">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4">Strategy</th>
            {universes.map((u) => (
              <th key={u} className="py-2 pr-4 font-mono">
                {u}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.strategy} className="border-b last:border-0">
              <td className="py-2 pr-4 font-mono">{row.strategy}</td>
              {universes.map((u) => {
                const leg = row.byUniverse.get(u);
                return (
                  <td key={u} className="py-2 pr-4">
                    {leg ? (
                      <StatusPill leg={leg} />
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
