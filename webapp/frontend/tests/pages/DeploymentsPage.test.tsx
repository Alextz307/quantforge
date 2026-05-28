import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import type { HoldoutEvalSummary } from "@/api/holdout";
import { API_PATHS, toMswPath } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { DeploymentsPage } from "@/pages/DeploymentsPage";
import { DEPLOY_SPY, HOLDOUT_DEMO_SUMMARY } from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.deployments} element={<DeploymentsPage />} />
      <Route path={ROUTES.deploymentDetail} element={<div>deployment detail</div>} />
    </Routes>
  );
}

describe("DeploymentsPage", () => {
  it("lists deployments returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.deployments] });
    expect(await screen.findByRole("link", { name: DEPLOY_SPY.name })).toBeInTheDocument();
  });

  it("opens the deploy picker and ranks holdout-backed models by Sharpe, nulls last", async () => {
    const NULL_EVAL: HoldoutEvalSummary = {
      name: "MomentumGatekeeper__qqq_daily_5y",
      store: "studies/main/holdout_evals",
      created_at: "2026-05-21T00:00:00Z",
      source_kind: "hpo",
      source_id: "MomentumGatekeeper__qqq_daily_5y",
      holdout_start: "2026-01-01T00:00:00Z",
      sharpe_ratio: null,
    };
    server.use(
      http.get(API_PATHS.holdoutEvals, () => HttpResponse.json([NULL_EVAL, HOLDOUT_DEMO_SUMMARY])),
    );
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.deployments] });
    await screen.findByRole("link", { name: DEPLOY_SPY.name });

    await user.click(screen.getByTestId("deployments-new-cta"));

    const table = await screen.findByTestId("deploy-picker-table");
    const bodyRows = within(table).getAllByRole("row").slice(1);
    // HOLDOUT_DEMO_SUMMARY carries Sharpe 1.42; NULL_EVAL has none (null sinks last).
    expect(bodyRows[0]).toHaveTextContent(HOLDOUT_DEMO_SUMMARY.name);
    expect(bodyRows[1]).toHaveTextContent(NULL_EVAL.name);
  });

  it("deploys a model from the picker and navigates to the new deployment", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.deployments] });
    await screen.findByRole("link", { name: DEPLOY_SPY.name });

    await user.click(screen.getByTestId("deployments-new-cta"));
    await user.click(
      await screen.findByTestId(
        `deploy-${HOLDOUT_DEMO_SUMMARY.source_kind}-${HOLDOUT_DEMO_SUMMARY.source_id}`,
      ),
    );

    expect(await screen.findByText("deployment detail")).toBeInTheDocument();
  });

  it("removes a deployment from the list after a confirmed delete", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    let deleted = false;
    server.use(
      http.get(API_PATHS.deployments, () => HttpResponse.json(deleted ? [] : [DEPLOY_SPY])),
      http.delete(toMswPath(API_PATHS.deployment), () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.deployments] });

    await screen.findByRole("link", { name: DEPLOY_SPY.name });
    await user.click(screen.getByRole("button", { name: `Delete ${DEPLOY_SPY.name}` }));
    await waitFor(
      () => {
        expect(screen.queryByRole("link", { name: DEPLOY_SPY.name })).not.toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });
});
