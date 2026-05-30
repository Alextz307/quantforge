import { useMemo } from "react";
import { useQueries } from "@tanstack/react-query";
import { apiClient } from "@/api/client";
import { extractApiError } from "@/api/errors";
import { API_PATHS } from "@/api/paths";
import { queryKeys } from "@/api/queryKeys";
import { type FoldRow } from "@/api/runs";
import { EquityChart, type EquityTrace } from "@/components/charts/EquityChart";

export interface OverlaySpec {
  label: string;
  experimentId: string;
}

interface OverlayProps {
  specs: readonly OverlaySpec[];
  height?: number;
}

// Walk-forward folds are temporally non-overlapping; each fold's equity_curve
// starts at 1.0 over its own test window. Chaining stitches them into a single
// continuous cumulative-equity curve so two strategies can be compared on the
// same y-axis. Without chaining, every fold would reset to 1.0 and the overlay
// would mislead the eye.
function chainFolds(folds: readonly FoldRow[]): number[] {
  const sorted = [...folds].sort((a, b) => a.fold_index - b.fold_index);
  const out: number[] = [];
  let running = 1;
  for (const fold of sorted) {
    if (fold.equity_curve.length === 0) continue;
    const start = fold.equity_curve[0];
    if (start === undefined || start === 0) continue;
    for (const v of fold.equity_curve) out.push(running * (v / start));
    const last = fold.equity_curve[fold.equity_curve.length - 1];
    if (last !== undefined) running = running * (last / start);
  }
  return out;
}

export function EquityOverlayChart({ specs, height = 420 }: OverlayProps) {
  const queries = useQueries({
    queries: specs.map((spec) => ({
      queryKey: queryKeys.runFolds(spec.experimentId),
      queryFn: async (): Promise<FoldRow[]> => {
        const { data, error, response } = await apiClient.GET(API_PATHS.runFolds, {
          params: { path: { experiment_id: spec.experimentId } },
        });
        if (!response.ok || !data) throw new Error(extractApiError(error, "Failed to load folds"));
        return data;
      },
      staleTime: Infinity,
    })),
  });

  const anyPending = queries.some((q) => q.isPending);
  // dataUpdatedAt advances only on real fetches, so this memo recomputes on
  // genuine data change but skips parent re-renders that don't touch the queries.
  const queryStamps = queries.map((q) => q.dataUpdatedAt).join("|");
  const { traces, failedLabels } = useMemo(() => {
    const t: EquityTrace[] = [];
    const failed: string[] = [];
    specs.forEach((spec, i) => {
      const q = queries[i];
      if (!q || q.isPending) return;
      if (q.isError) {
        failed.push(spec.label);
        return;
      }
      const equity = chainFolds(q.data);
      if (equity.length > 0) t.push({ name: spec.label, equity });
    });
    return { traces: t, failedLabels: failed };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specs, queryStamps]);

  if (anyPending && traces.length === 0) {
    return (
      <div data-testid="equity-overlay-loading" className="text-sm text-muted-foreground py-8">
        Loading equity curves...
      </div>
    );
  }

  return (
    <div data-testid="equity-overlay" data-trace-count={traces.length}>
      <EquityChart
        traces={traces}
        height={height}
        xLabel="Walk-forward bar"
        yLabel="Cumulative equity"
      />
      {failedLabels.length > 0 && (
        <p
          data-testid="equity-overlay-failed-labels"
          className="text-xs text-destructive mt-2 font-mono"
        >
          Could not load: {failedLabels.join(", ")}
        </p>
      )}
    </div>
  );
}
