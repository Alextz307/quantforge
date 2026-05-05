import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-cartesian-dist-min";
import type { Data, Layout } from "plotly.js";
import type { FoldRow } from "@/api/runs";

const Plot = createPlotlyComponent(Plotly);

export interface EquityChartProps {
  folds: readonly FoldRow[];
  height?: number;
}

export function EquityChart({ folds, height = 400 }: EquityChartProps) {
  const traces = useMemo<Data[]>(
    () =>
      folds.map((f) => ({
        x: f.equity_curve.map((_, i) => i),
        y: f.equity_curve,
        name: `Fold ${String(f.fold_index)}`,
        type: "scatter",
        mode: "lines",
      })),
    [folds],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 60, r: 20, t: 20, b: 60 },
      xaxis: { title: { text: "Bar within fold" } },
      yaxis: { title: { text: "Equity (cumulative)" } },
      legend: { orientation: "h", x: 0, y: -0.2 },
      showlegend: true,
    }),
    [height],
  );

  if (folds.length === 0) {
    return (
      <div
        data-testid="equity-chart-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        No fold equity data available.
      </div>
    );
  }

  return (
    <div data-testid="equity-chart" data-fold-count={folds.length}>
      <Plot
        data={traces}
        layout={layout}
        config={{ displayModeBar: true, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
