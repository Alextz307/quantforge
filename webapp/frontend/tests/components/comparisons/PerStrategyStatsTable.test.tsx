import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { PerStrategyStatsTable } from "@/components/comparisons/PerStrategyStatsTable";
import { COMPARISON_DEMO_DETAIL, RUN_SPY } from "../../msw/handlers";
import { ROUTER_FUTURE_FLAGS } from "../../util/router";

function renderTable(rows: typeof COMPARISON_DEMO_DETAIL.per_strategy_stats) {
  render(
    <MemoryRouter future={ROUTER_FUTURE_FLAGS}>
      <PerStrategyStatsTable rows={rows} />
    </MemoryRouter>,
  );
}

describe("PerStrategyStatsTable", () => {
  it("renders an empty message when there are no rows", () => {
    renderTable([]);
    expect(screen.getByText(/no per-strategy stats/i)).toBeInTheDocument();
  });

  it("renders a row per strategy and links each strategy to its run", () => {
    renderTable(COMPARISON_DEMO_DETAIL.per_strategy_stats);
    const link = screen.getByRole("link", { name: RUN_SPY.strategy });
    expect(link).toHaveAttribute("href", `/runs/${RUN_SPY.experiment_id}`);
  });
});
