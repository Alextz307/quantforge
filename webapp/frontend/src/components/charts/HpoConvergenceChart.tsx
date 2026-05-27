import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { Plot, useThemedLayout } from "@/components/charts/plot";
import type { StudyDirection, TrialRow } from "@/api/hpo";
import { TRIAL_STATE_COMPLETE } from "@/lib/trialState";

interface ConvergencePoint {
  number: number;
  value: number;
  best: number;
}

function computeConvergence(
  trials: readonly TrialRow[],
  direction: StudyDirection,
): ConvergencePoint[] {
  const completed = trials
    .filter((t) => t.state === TRIAL_STATE_COMPLETE && t.value !== null)
    .map((t) => ({ number: t.number, value: t.value as number }))
    .sort((a, b) => a.number - b.number);
  const out: ConvergencePoint[] = [];
  let best: number | null = null;
  for (const c of completed) {
    if (best === null) {
      best = c.value;
    } else if (direction === "maximize") {
      best = Math.max(best, c.value);
    } else {
      best = Math.min(best, c.value);
    }
    out.push({ number: c.number, value: c.value, best });
  }
  return out;
}

export interface HpoConvergenceChartProps {
  trials: readonly TrialRow[];
  direction: StudyDirection;
  height?: number;
}

export function HpoConvergenceChart({ trials, direction, height = 360 }: HpoConvergenceChartProps) {
  const points = useMemo(() => computeConvergence(trials, direction), [trials, direction]);

  const plotData = useMemo<Data[]>(() => {
    const xs = points.map((p) => p.number);
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "Trial value",
        x: xs,
        y: points.map((p) => p.value),
        marker: { size: 6 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: `Best so far (${direction})`,
        x: xs,
        y: points.map((p) => p.best),
        line: { width: 2 },
      },
    ];
  }, [points, direction]);

  const baseLayout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 60, r: 20, t: 20, b: 60 },
      xaxis: { title: { text: "Trial #" } },
      yaxis: { title: { text: "Objective value" } },
      legend: { orientation: "h", x: 0, y: -0.2 },
      showlegend: true,
    }),
    [height],
  );
  const layout = useThemedLayout(baseLayout);

  if (points.length === 0) {
    return (
      <div
        data-testid="hpo-convergence-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        No completed trials yet.
      </div>
    );
  }

  return (
    <div data-testid="hpo-convergence" data-trial-count={points.length}>
      <Plot
        data={plotData}
        layout={layout}
        config={{ displayModeBar: true, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
