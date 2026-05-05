import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RunDetailPage } from "@/pages/RunDetailPage";
import { ROUTES, runDetailPath } from "@/lib/routes";
import { RUN_SPY, RUN_SPY_DETAIL, RUN_SPY_FOLDS } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.runs} element={<div>runs list</div>} />
      <Route path={ROUTES.runDetail} element={<RunDetailPage />} />
    </Routes>
  );
}

describe("RunDetailPage", () => {
  it("renders manifest fields, fold metrics, equity chart, and plot links", async () => {
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByText(RUN_SPY_DETAIL.name)).toBeInTheDocument();
    expect(screen.getByText(RUN_SPY_DETAIL.experiment_id)).toBeInTheDocument();
    expect(screen.getByText(RUN_SPY_DETAIL.strategy)).toBeInTheDocument();

    const chart = await screen.findByTestId("equity-chart");
    expect(chart).toHaveAttribute("data-fold-count", String(RUN_SPY_FOLDS.length));

    const plotLink = screen.getByRole("link", { name: "equity.png" });
    expect(plotLink).toHaveAttribute("href", `/api/runs/${RUN_SPY.experiment_id}/plots/equity.png`);
  });
});
