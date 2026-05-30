import { useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { Plot, useThemedLayout } from "@/components/charts/plot";
import { buildFeatureImportanceBars } from "@/components/charts/featureImportanceBars";
import { Button } from "@/components/ui/button";
import type { FeatureImportanceResponse, ImportanceMethod } from "@/api/runs";

export interface FeatureImportanceChartProps {
  response: FeatureImportanceResponse;
  height?: number;
}

const METHOD_ORDER = ["permutation", "xgb_gain"] as const satisfies readonly ImportanceMethod[];

const METHOD_LABEL: Record<ImportanceMethod, string> = {
  permutation: "Permutation (OOS drop)",
  xgb_gain: "XGBoost gain",
};

const METHOD_AXIS_TITLE: Record<ImportanceMethod, string> = {
  permutation: "Mean out-of-sample score drop",
  xgb_gain: "Average gain",
};

const METHOD_COLOR: Record<ImportanceMethod, string> = {
  permutation: "#3b82f6",
  xgb_gain: "#a855f7",
};

export function FeatureImportanceChart({ response, height = 360 }: FeatureImportanceChartProps) {
  // A method needs at least one finite bar to be selectable, so the default tab
  // never lands on a method whose aggregates are all null.
  const availableMethods = useMemo(
    () =>
      METHOD_ORDER.filter((m) =>
        response.entries.some((e) => e.method === m && e.importance !== null),
      ),
    [response.entries],
  );
  const [selected, setSelected] = useState<ImportanceMethod>(METHOD_ORDER[0]);
  const method = availableMethods.includes(selected)
    ? selected
    : (availableMethods[0] ?? METHOD_ORDER[0]);

  const bars = useMemo(
    () => buildFeatureImportanceBars(response.entries, method),
    [response.entries, method],
  );

  const plotData = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: bars.values,
        y: bars.features,
        marker: { color: METHOD_COLOR[method] },
        ...(bars.errors !== null
          ? { error_x: { type: "data", array: bars.errors, visible: true } }
          : {}),
        hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
      },
    ],
    [bars, method],
  );

  const baseLayout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 150, r: 20, t: 20, b: 40 },
      // Permutation importance can be negative (shuffling a feature can improve
      // the OOS score), so leave the x-range to autorange and keep the zero line.
      xaxis: { title: { text: METHOD_AXIS_TITLE[method] }, zeroline: true },
      yaxis: { automargin: true },
    }),
    [height, method],
  );
  const layout = useThemedLayout(baseLayout);

  if (availableMethods.length === 0) {
    const message = response.message ?? "No feature importance for this run.";
    return (
      <div
        data-testid="feature-importance-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        {message}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {availableMethods.length > 1 && (
        <div className="flex gap-2" role="group" aria-label="Importance method">
          {availableMethods.map((m) => (
            <Button
              key={m}
              type="button"
              size="sm"
              variant={m === method ? "default" : "outline"}
              aria-pressed={m === method}
              data-testid={`feature-importance-method-${m}`}
              onClick={() => {
                setSelected(m);
              }}
            >
              {METHOD_LABEL[m]}
            </Button>
          ))}
        </div>
      )}
      <div
        data-testid="feature-importance"
        data-feature-count={bars.features.length}
        data-method={method}
      >
        <Plot
          data={plotData}
          layout={layout}
          config={{ displayModeBar: true, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
