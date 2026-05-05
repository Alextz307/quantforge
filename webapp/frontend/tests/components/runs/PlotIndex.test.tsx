import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PlotIndex } from "@/components/runs/PlotIndex";

describe("PlotIndex", () => {
  it("renders an empty message when there are no plots", () => {
    render(<PlotIndex experimentId="exp_x" plots={[]} />);
    expect(screen.getByText(/no static figures/i)).toBeInTheDocument();
  });

  it("renders a download link per plot pointing at the right backend path", () => {
    render(<PlotIndex experimentId="exp_x" plots={["equity.png", "fold_stability.svg"]} />);
    const equity = screen.getByRole("link", { name: "equity.png" });
    expect(equity).toHaveAttribute("href", "/api/runs/exp_x/plots/equity.png");
    expect(equity).toHaveAttribute("download", "equity.png");
    expect(screen.getByRole("link", { name: "fold_stability.svg" })).toHaveAttribute(
      "href",
      "/api/runs/exp_x/plots/fold_stability.svg",
    );
  });
});
