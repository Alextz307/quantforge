import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EquityChart, type EquityTrace } from "@/components/charts/EquityChart";
import { renderWithProviders } from "../../util/render";

const TWO_TRACES: EquityTrace[] = [
  { name: "Fold 0", equity: [1, 1.01, 1.02] },
  { name: "Fold 1", equity: [1, 0.99, 1.01] },
];

describe("EquityChart", () => {
  it("renders an empty-state when no traces are supplied", () => {
    renderWithProviders(<EquityChart traces={[]} />);
    expect(screen.getByTestId("equity-chart-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("equity-chart")).not.toBeInTheDocument();
  });

  it("forwards one Plotly trace per input trace", () => {
    renderWithProviders(<EquityChart traces={TWO_TRACES} />);
    const wrapper = screen.getByTestId("equity-chart");
    expect(wrapper).toHaveAttribute("data-trace-count", String(TWO_TRACES.length));
    const plot = screen.getByTestId("plotly-plot");
    expect(plot).toHaveAttribute("data-trace-count", String(TWO_TRACES.length));
    expect(plot).toHaveAttribute("data-trace-names", "Fold 0,Fold 1");
  });

  it("reserves a fixed, clipped box so a mid-resize plot can't overlap siblings", () => {
    const HEIGHT = 260;
    renderWithProviders(<EquityChart traces={TWO_TRACES} height={HEIGHT} />);
    const wrapper = screen.getByTestId("equity-chart");
    // Height is CSS-fixed (not Plotly-driven) so the container's width is known
    // at first paint; overflow:hidden clips any degenerate intermediate draw
    // instead of letting the absolute main-svg paint over the table below.
    expect(wrapper).toHaveStyle({
      height: `${String(HEIGHT)}px`,
      overflow: "hidden",
      position: "relative",
    });
  });
});
