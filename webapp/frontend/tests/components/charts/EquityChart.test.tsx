import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EquityChart } from "@/components/charts/EquityChart";
import { RUN_SPY_FOLDS } from "../../msw/handlers";

describe("EquityChart", () => {
  it("renders an empty-state when no folds are supplied", () => {
    render(<EquityChart folds={[]} />);
    expect(screen.getByTestId("equity-chart-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("equity-chart")).not.toBeInTheDocument();
  });

  it("forwards one trace per fold to the underlying Plot component", () => {
    render(<EquityChart folds={RUN_SPY_FOLDS} />);
    const wrapper = screen.getByTestId("equity-chart");
    expect(wrapper).toHaveAttribute("data-fold-count", String(RUN_SPY_FOLDS.length));
    const plot = screen.getByTestId("plotly-plot");
    expect(plot).toHaveAttribute("data-trace-count", String(RUN_SPY_FOLDS.length));
    expect(plot).toHaveAttribute("data-trace-names", "Fold 0,Fold 1");
  });
});
