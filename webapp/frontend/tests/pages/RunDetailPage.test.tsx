import { screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { API_PATHS, toMswPath } from "@/api/paths";
import { RunDetailPage } from "@/pages/RunDetailPage";
import { ROUTES, runDetailPath } from "@/lib/routes";
import { RUN_SPY, RUN_SPY_DETAIL, RUN_SPY_FOLDS } from "../msw/handlers";
import { server } from "../msw/server";
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
    expect(chart).toHaveAttribute("data-trace-count", String(RUN_SPY_FOLDS.length));

    const plotLink = screen.getByRole("link", { name: "equity.png" });
    expect(plotLink).toHaveAttribute("href", `/api/runs/${RUN_SPY.experiment_id}/plots/equity.png`);
  });

  it("explains the empty plot index when the run is nested inside a comparison", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.run), () =>
        HttpResponse.json({
          ...RUN_SPY_DETAIL,
          store: "thesis_demo/comparisons/pipeline_compare",
          plots: [],
        }),
      ),
    );

    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByText(/nested inside a comparison/i)).toBeInTheDocument();
    expect(screen.queryByTestId("plot-index")).not.toBeInTheDocument();
  });

  it("explains the empty plot index when the run is an HPO trial", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.run), () =>
        HttpResponse.json({
          ...RUN_SPY_DETAIL,
          store: "studies/main/hpo/AdaptiveBollinger__spy_daily_5y/trials_artifacts",
          plots: [],
        }),
      ),
    );

    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByText(/individual HPO trial/i)).toBeInTheDocument();
    expect(screen.queryByTestId("plot-index")).not.toBeInTheDocument();
  });
});
