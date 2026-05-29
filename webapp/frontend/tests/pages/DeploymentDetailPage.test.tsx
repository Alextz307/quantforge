import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { API_PATHS, toMswPath } from "@/api/paths";
import { ROUTES, deploymentDetailPath } from "@/lib/routes";
import { DeploymentDetailPage } from "@/pages/DeploymentDetailPage";
import { DEPLOY_SPY, DEPLOY_SPY_DETAIL } from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.deploymentDetail} element={<DeploymentDetailPage />} />
    </Routes>
  );
}

function renderDetail() {
  return renderWithProviders(<Tree />, {
    initialEntries: [deploymentDetailPath(DEPLOY_SPY.id)],
  });
}

describe("DeploymentDetailPage", () => {
  it("renders the deployment name and source", async () => {
    renderDetail();
    expect(await screen.findByTestId("deployment-name")).toHaveTextContent(DEPLOY_SPY.name);
    expect(screen.getByText(new RegExp(DEPLOY_SPY.source_id))).toBeInTheDocument();
  });

  it("computes today's signal on mount and renders the history table", async () => {
    renderDetail();
    expect(await screen.findByText(/LONG/)).toBeInTheDocument();
    const table = await screen.findByTestId("signal-history-table");
    expect(within(table).getAllByTestId("signal-badge").length).toBeGreaterThan(0);
  });

  it("renders the signal performance summary and per-signal scores", async () => {
    renderDetail();
    const perf = await screen.findByTestId("signal-performance");
    expect(within(perf).getByText("100.00%")).toBeInTheDocument(); // hit rate 1.0
    expect(within(perf).getByText("1 / 3")).toBeInTheDocument(); // scored count
  });

  it("shows gross and net cumulative with a cost-tier selector", async () => {
    renderDetail();
    const perf = await screen.findByTestId("signal-performance");
    expect(within(perf).getByText("1.20%")).toBeInTheDocument(); // gross cumulative
    expect(within(perf).getByText("1.16%")).toBeInTheDocument(); // net of costs
    expect(screen.getByTestId("cost-tier-selector")).toBeInTheDocument();
  });

  it("distinguishes scored, holding, and pending signals in the history", async () => {
    renderDetail();
    const table = await screen.findByTestId("signal-history-table");
    // scored row → outcome; holding row → live; pending row → not entered.
    expect(within(table).getByText("✓ win")).toBeInTheDocument();
    expect(within(table).getByText("holding")).toBeInTheDocument();
    expect(within(table).getByText("pending")).toBeInTheDocument();
    // the holding row surfaces its entry-open price even with no score yet
    expect(within(table).getByText("101.20")).toBeInTheDocument();
  });

  it("surfaces a 422 predict error inline", async () => {
    server.use(
      http.post(toMswPath(API_PATHS.deploymentPredict), () =>
        HttpResponse.json(
          {
            detail: "Model trained through 2026-05-28; first valid prediction date is 2026-05-29.",
          },
          { status: 422 },
        ),
      ),
    );
    renderDetail();
    expect(await screen.findByText(/first valid prediction date/)).toBeInTheDocument();
  });

  it("renames the deployment and reflects the new name", async () => {
    // Mirror the backend: a rename persists, so the refetch triggered by
    // cache invalidation returns the new name rather than the seed.
    let currentName = DEPLOY_SPY.name;
    server.use(
      http.patch(toMswPath(API_PATHS.deployment), async ({ request }) => {
        const body = (await request.json()) as { name: string };
        currentName = body.name;
        return HttpResponse.json({ ...DEPLOY_SPY_DETAIL, name: currentName });
      }),
      http.get(toMswPath(API_PATHS.deployment), () =>
        HttpResponse.json({ ...DEPLOY_SPY_DETAIL, name: currentName }),
      ),
    );
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("deployment-name");

    await user.click(screen.getByTestId("deployment-rename-cta"));
    const input = screen.getByTestId("deployment-rename-input");
    await user.clear(input);
    await user.type(input, "Renamed deployment");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Renamed deployment")).toBeInTheDocument();
  });
});
