import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ComparisonDetailPage } from "@/pages/ComparisonDetailPage";
import { ROUTES, comparisonDetailPath } from "@/lib/routes";
import {
  COMPARISON_DEMO_DETAIL,
  COMPARISON_DEMO_SUMMARY,
  RUN_IVV_VOO,
  RUN_SPY,
} from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.comparisons} element={<div>list</div>} />
      <Route path={ROUTES.comparisonDetail} element={<ComparisonDetailPage />} />
    </Routes>
  );
}

describe("ComparisonDetailPage", () => {
  it("renders identity, per-strategy stats, equity overlay, and plot links", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [comparisonDetailPath(COMPARISON_DEMO_SUMMARY.name)],
    });

    expect(await screen.findByText(COMPARISON_DEMO_DETAIL.name)).toBeInTheDocument();

    const table = await screen.findByTestId("per-strategy-stats-table");
    expect(table).toBeInTheDocument();

    const overlay = await screen.findByTestId("equity-overlay");
    expect(overlay).toHaveAttribute(
      "data-trace-count",
      String(COMPARISON_DEMO_DETAIL.per_strategy_stats.length),
    );

    const plotLink = screen.getByRole("link", { name: "ranking.png" });
    expect(plotLink).toHaveAttribute(
      "href",
      `/api/comparisons/${COMPARISON_DEMO_SUMMARY.name}/plots/ranking.png`,
    );
  });

  it("links each strategy row to its underlying run", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [comparisonDetailPath(COMPARISON_DEMO_SUMMARY.name)],
    });

    const spyLink = await screen.findByRole("link", { name: RUN_SPY.strategy });
    expect(spyLink).toHaveAttribute("href", `/runs/${RUN_SPY.experiment_id}`);
    const pairsLink = screen.getByRole("link", { name: RUN_IVV_VOO.strategy });
    expect(pairsLink).toHaveAttribute("href", `/runs/${RUN_IVV_VOO.experiment_id}`);
  });
});
