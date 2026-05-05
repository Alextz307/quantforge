import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PlotIndex } from "@/components/PlotIndex";

const URL_FOR = (plotName: string) => `/api/runs/exp_x/plots/${plotName}`;

describe("PlotIndex", () => {
  it("renders an empty message when there are no plots", () => {
    render(<PlotIndex plots={[]} urlForPlot={URL_FOR} />);
    expect(screen.getByText(/no static figures/i)).toBeInTheDocument();
  });

  it("renders a download link per plot using urlForPlot", () => {
    render(<PlotIndex plots={["equity.png", "fold_stability.svg"]} urlForPlot={URL_FOR} />);
    const equity = screen.getByRole("link", { name: "equity.png" });
    expect(equity).toHaveAttribute("href", "/api/runs/exp_x/plots/equity.png");
    expect(equity).toHaveAttribute("download", "equity.png");
    expect(screen.getByRole("link", { name: "fold_stability.svg" })).toHaveAttribute(
      "href",
      "/api/runs/exp_x/plots/fold_stability.svg",
    );
  });

  it("respects a custom emptyMessage", () => {
    render(<PlotIndex plots={[]} urlForPlot={URL_FOR} emptyMessage="No comparison figures." />);
    expect(screen.getByText("No comparison figures.")).toBeInTheDocument();
  });
});
