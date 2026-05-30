import { Link } from "react-router-dom";
import type { TrialRow } from "@/api/hpo";
import { formatDateTime, formatMetric } from "@/lib/format";
import { runDetailPath } from "@/lib/routes";
import { TRIAL_STATE_COMPLETE, trialStateStyle } from "@/lib/trialState";

function StatePill({ state }: { state: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${trialStateStyle(state)}`}
    >
      {state.toLowerCase()}
    </span>
  );
}

function formatParamValue(v: unknown): string {
  if (typeof v === "number") return formatMetric(v);
  if (v === null || v === undefined) return "null";
  if (typeof v === "string" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

function paramsSummary(params: Readonly<Record<string, unknown>>): string {
  const keys = Object.keys(params).sort();
  if (keys.length === 0) return "-";
  return keys.map((k) => `${k}=${formatParamValue(params[k])}`).join(", ");
}

export interface TrialTableProps {
  trials: readonly TrialRow[];
  bestTrialNumber: number | null;
}

export function TrialTable({ trials, bestTrialNumber }: TrialTableProps) {
  if (trials.length === 0) {
    return <p className="text-sm text-muted-foreground">No trials recorded.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="trial-table">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4 text-right">#</th>
            <th className="py-2 pr-4">State</th>
            <th className="py-2 pr-4 text-right">Value</th>
            <th className="py-2 pr-4">Params</th>
            <th className="py-2 pr-4">Started</th>
            <th className="py-2 pr-0">Run</th>
          </tr>
        </thead>
        <tbody>
          {trials.map((t) => {
            const isBest = bestTrialNumber !== null && t.number === bestTrialNumber;
            return (
              <tr
                key={t.number}
                className={`border-b last:border-0 ${isBest ? "bg-primary/5" : ""}`}
                data-testid={`trial-row-${String(t.number)}`}
              >
                <td className="py-2 pr-4 text-right font-mono">{t.number}</td>
                <td className="py-2 pr-4">
                  <StatePill state={t.state} />
                </td>
                <td className="py-2 pr-4 text-right font-mono">{formatMetric(t.value)}</td>
                <td className="py-2 pr-4 font-mono text-xs">{paramsSummary(t.params)}</td>
                <td className="py-2 pr-4 font-mono text-xs">
                  {t.datetime_start ? formatDateTime(t.datetime_start) : "-"}
                </td>
                <td className="py-2 pr-0 font-mono text-xs">
                  {t.experiment_id && t.state === TRIAL_STATE_COMPLETE ? (
                    <Link
                      to={runDetailPath(t.experiment_id)}
                      className="text-primary hover:underline"
                    >
                      open
                    </Link>
                  ) : (
                    "-"
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
