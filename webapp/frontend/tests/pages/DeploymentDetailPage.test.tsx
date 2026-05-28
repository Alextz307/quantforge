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
