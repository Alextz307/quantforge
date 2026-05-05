import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { TrialTable } from "@/components/hpo/TrialTable";
import { HPO_DEMO_DETAIL, HPO_DEMO_TRIALS } from "../../msw/handlers";
import { ROUTER_FUTURE_FLAGS } from "../../util/router";

function renderTable() {
  render(
    <MemoryRouter future={ROUTER_FUTURE_FLAGS}>
      <TrialTable trials={HPO_DEMO_TRIALS} bestTrialNumber={HPO_DEMO_DETAIL.best_trial_number} />
    </MemoryRouter>,
  );
}

describe("TrialTable", () => {
  it("renders an empty state when there are no trials", () => {
    render(
      <MemoryRouter future={ROUTER_FUTURE_FLAGS}>
        <TrialTable trials={[]} bestTrialNumber={null} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/no trials recorded/i)).toBeInTheDocument();
  });

  it("renders one row per trial", () => {
    renderTable();
    for (const t of HPO_DEMO_TRIALS) {
      expect(screen.getByTestId(`trial-row-${String(t.number)}`)).toBeInTheDocument();
    }
  });

  it("highlights the best trial row with the primary background", () => {
    renderTable();
    const bestRow = screen.getByTestId(
      `trial-row-${String(HPO_DEMO_DETAIL.best_trial_number)}`,
    );
    expect(bestRow.className).toMatch(/bg-primary/);
  });

  it("links completed trials with experiment_id back to the run", () => {
    renderTable();
    const completedWithExp = HPO_DEMO_TRIALS.find(
      (t) => t.state === "COMPLETE" && t.experiment_id !== null,
    );
    if (!completedWithExp) throw new Error("expected a completed trial with experiment_id");
    const link = screen.getAllByRole("link", { name: "open" })[0];
    expect(link).toHaveAttribute(
      "href",
      `/runs/${completedWithExp.experiment_id ?? ""}`,
    );
  });
});
