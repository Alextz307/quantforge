import { screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RunDetailPage } from "@/pages/RunDetailPage";
import { API_PATHS, toMswPath } from "@/api/paths";
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

  it("shows the 'Run holdout eval' CTA when the manifest has a holdout boundary", async () => {
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });
    const cta = await screen.findByTestId("run-detail-holdout-cta");
    expect(cta).toHaveAttribute(
      "href",
      `${ROUTES.configureHoldout}?source_kind=run&source_id=${RUN_SPY.experiment_id}`,
    );
  });

  it("hides the holdout CTA when the manifest has no holdout boundary", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.run), ({ params }) =>
        HttpResponse.json({
          ...RUN_SPY_DETAIL,
          experiment_id: String(params.experiment_id),
          holdout_start: null,
        }),
      ),
    );
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });
    await screen.findByText(RUN_SPY_DETAIL.name);
    expect(screen.queryByTestId("run-detail-holdout-cta")).not.toBeInTheDocument();
  });
});
