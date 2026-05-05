import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { Plot } from "@/components/charts/plot";
import type { PerRegimeStatsRow } from "@/api/regime";

const METRIC_KEYS = [
  "sharpe_mean",
  "sortino_mean",
  "calmar_mean",
  "total_return_mean",
  "max_drawdown_mean",
  "win_rate_mean",
] as const satisfies readonly (keyof PerRegimeStatsRow)[];

const METRIC_LABELS: Record<(typeof METRIC_KEYS)[number], string> = {
  sharpe_mean: "Sharpe",
  sortino_mean: "Sortino",
  calmar_mean: "Calmar",
  total_return_mean: "Total return",
  max_drawdown_mean: "Max drawdown",
  win_rate_mean: "Win rate",
};

const Y_LABELS: string[] = METRIC_KEYS.map((k) => METRIC_LABELS[k]);

export interface RegimeMetricHeatmapProps {
  rows: readonly PerRegimeStatsRow[];
  height?: number;
}

export function RegimeMetricHeatmap({ rows, height = 360 }: RegimeMetricHeatmapProps) {
  const { z, x } = useMemo(() => {
    const labels = rows.map((r) => r.regime_label);
    const matrix = METRIC_KEYS.map((key) => rows.map((r) => r[key]));
    return { z: matrix, x: labels };
  }, [rows]);

  const plotData = useMemo<Data[]>(
    () => [
      {
        type: "heatmap",
        z,
        x,
        y: Y_LABELS,
        colorscale: "RdBu",
        zmid: 0,
        hovertemplate: "%{y}<br>%{x}: %{z:.3f}<extra></extra>",
      },
    ],
    [z, x],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 120, r: 20, t: 20, b: 60 },
      xaxis: { title: { text: "Regime" } },
      yaxis: { title: { text: "Metric" }, automargin: true },
    }),
    [height],
  );

  if (rows.length === 0) {
    return (
      <div
        data-testid="regime-heatmap-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        No per-regime stats to render.
      </div>
    );
  }

  return (
    <div data-testid="regime-heatmap" data-regime-count={rows.length}>
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
