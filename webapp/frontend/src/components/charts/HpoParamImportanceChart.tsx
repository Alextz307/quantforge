import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { Plot } from "@/components/charts/plot";
import { useThemedLayout } from "@/components/charts/chartTheme";
import { buildHpoImportanceRows } from "@/components/charts/hpoImportanceRows";
import type { ParamImportanceResponse } from "@/api/hpo";

export interface HpoParamImportanceChartProps {
  response: ParamImportanceResponse;
  height?: number;
}

export function HpoParamImportanceChart({ response, height = 320 }: HpoParamImportanceChartProps) {
  const rows = useMemo(() => buildHpoImportanceRows(response.importance), [response.importance]);

  const plotData = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: rows.map((r) => r.value),
        y: rows.map((r) => r.name),
        marker: { color: "#3b82f6" },
        hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
      },
    ],
    [rows],
  );

  const baseLayout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 140, r: 20, t: 20, b: 40 },
      xaxis: { title: { text: "Relative importance" }, range: [0, 1] },
      yaxis: { automargin: true },
    }),
    [height],
  );
  const layout = useThemedLayout(baseLayout);

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
      <p className="text-xs text-muted-foreground" data-testid="hpo-importance-note">
        Optuna fANOVA over completed trials: the share of validation-objective variance each
        hyperparameter explains across the searched ranges &mdash; which knobs moved the result most
        during the search, not the single best value.
      </p>
    </div>
  );
}
