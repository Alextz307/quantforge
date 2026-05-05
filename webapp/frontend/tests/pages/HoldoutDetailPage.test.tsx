import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { HoldoutDetailPage } from "@/pages/HoldoutDetailPage";
import { ROUTES, holdoutDetailPath } from "@/lib/routes";
import { HOLDOUT_DEMO_DETAIL, HOLDOUT_DEMO_SUMMARY, RUN_SPY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.holdout} element={<div>list</div>} />
      <Route path={ROUTES.holdoutDetail} element={<HoldoutDetailPage />} />
    </Routes>
  );
}

describe("HoldoutDetailPage", () => {
  it("renders identity, dev/holdout panel, equity chart, and plot links", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [holdoutDetailPath(HOLDOUT_DEMO_SUMMARY.name)],
    });

    expect(await screen.findByText(HOLDOUT_DEMO_DETAIL.name)).toBeInTheDocument();
    expect(await screen.findByTestId("dev-vs-holdout-panel")).toBeInTheDocument();

    const chart = await screen.findByTestId("equity-chart");
    expect(chart).toHaveAttribute("data-trace-count", "1");

    const plotLink = screen.getByRole("link", { name: "holdout_equity.png" });
    expect(plotLink).toHaveAttribute(
      "href",
      `/api/holdout-evals/${HOLDOUT_DEMO_SUMMARY.name}/plots/holdout_equity.png`,
    );
  });

  it("links the dev source back to the underlying run when source_kind is 'run'", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [holdoutDetailPath(HOLDOUT_DEMO_SUMMARY.name)],
    });

    const sourceLink = await screen.findByRole("link", { name: RUN_SPY.experiment_id });
    expect(sourceLink).toHaveAttribute("href", `/runs/${RUN_SPY.experiment_id}`);
  });
});
