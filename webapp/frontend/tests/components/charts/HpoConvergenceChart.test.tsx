import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HpoConvergenceChart } from "@/components/charts/HpoConvergenceChart";
import { HPO_DEMO_TRIALS } from "../../msw/handlers";

const COMPLETED_TRIAL_COUNT = HPO_DEMO_TRIALS.filter(
  (t) => t.state === "COMPLETE" && t.value !== null,
).length;

describe("HpoConvergenceChart", () => {
  it("renders an empty state when no trials are completed", () => {
    render(<HpoConvergenceChart trials={[]} direction="maximize" />);
    expect(screen.getByTestId("hpo-convergence-empty")).toBeInTheDocument();
  });

  it("renders the chart wrapper with the count of completed trials", () => {
    render(<HpoConvergenceChart trials={HPO_DEMO_TRIALS} direction="maximize" />);
    const wrapper = screen.getByTestId("hpo-convergence");
    expect(wrapper.getAttribute("data-trial-count")).toBe(String(COMPLETED_TRIAL_COUNT));
  });
});
