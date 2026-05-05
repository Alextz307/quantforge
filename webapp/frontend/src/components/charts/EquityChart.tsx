import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-cartesian-dist-min";
import type { Data, Layout } from "plotly.js";

const Plot = createPlotlyComponent(Plotly);

export interface EquityTrace {
  name: string;
  equity: readonly number[];
}

export interface EquityChartProps {
  traces: readonly EquityTrace[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
}

export function EquityChart({
  traces,
  height = 400,
  xLabel = "Bar",
  yLabel = "Equity (cumulative)",
}: EquityChartProps) {
  const plotData = useMemo<Data[]>(
    () =>
      traces.map((t) => ({
        x: t.equity.map((_, i) => i),
        y: [...t.equity],
        name: t.name,
        type: "scatter",
        mode: "lines",
      })),
    [traces],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 60, r: 20, t: 20, b: 60 },
      xaxis: { title: { text: xLabel } },
      yaxis: { title: { text: yLabel } },
      legend: { orientation: "h", x: 0, y: -0.2 },
      showlegend: true,
    }),
    [height, xLabel, yLabel],
  );

  if (traces.length === 0) {
    return (
      <div
        data-testid="equity-chart-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        No equity data available.
      </div>
    );
  }

  return (
    <div data-testid="equity-chart" data-trace-count={traces.length}>
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
