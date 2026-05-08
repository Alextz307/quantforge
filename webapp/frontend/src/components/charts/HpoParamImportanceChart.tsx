import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { Plot } from "@/components/charts/plot";
import type { ParamImportanceResponse } from "@/api/hpo";

export interface HpoParamImportanceChartProps {
  response: ParamImportanceResponse;
  height?: number;
}

interface ImportanceRow {
  name: string;
  value: number;
}

function sortDescending(importance: Record<string, number>): ImportanceRow[] {
  return Object.entries(importance)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
}

export function HpoParamImportanceChart({ response, height = 320 }: HpoParamImportanceChartProps) {
  const rows = useMemo(() => sortDescending(response.importance), [response.importance]);

  const plotData = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: rows.map((r) => r.value),
        // Plotly draws categorical y bottom-up, so reverse so the highest
        // importance sits at the top of the chart.
        y: rows.map((r) => r.name).reverse(),
        marker: { color: "#3b82f6" },
        hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
      },
    ],
    [rows],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 140, r: 20, t: 20, b: 40 },
      xaxis: { title: { text: "Relative importance" }, range: [0, 1] },
      yaxis: { automargin: true },
    }),
    [height],
  );

  if (rows.length === 0) {
    const message = response.message ?? "No importance data yet.";
    return (
      <div
        data-testid="hpo-importance-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        {message}
      </div>
    );
  }

  return (
    <div data-testid="hpo-importance" data-param-count={rows.length}>
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
