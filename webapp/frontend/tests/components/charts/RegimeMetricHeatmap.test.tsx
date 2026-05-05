import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RegimeMetricHeatmap } from "@/components/charts/RegimeMetricHeatmap";
import { REGIME_DEMO_DETAIL } from "../../msw/handlers";

describe("RegimeMetricHeatmap", () => {
  it("renders an empty state when no rows are provided", () => {
    render(<RegimeMetricHeatmap rows={[]} />);
    expect(screen.getByTestId("regime-heatmap-empty")).toBeInTheDocument();
  });

  it("renders the chart wrapper with a regime count", () => {
    render(<RegimeMetricHeatmap rows={REGIME_DEMO_DETAIL.per_regime_stats} />);
    const wrapper = screen.getByTestId("regime-heatmap");
    expect(wrapper.getAttribute("data-regime-count")).toBe(
      String(REGIME_DEMO_DETAIL.per_regime_stats.length),
    );
  });
});
