import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { HoldoutPage } from "@/pages/HoldoutPage";
import { ROUTES } from "@/lib/routes";
import { HOLDOUT_DEMO_SUMMARY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.holdout} element={<HoldoutPage />} />
      <Route path={ROUTES.holdoutDetail} element={<div>holdout detail</div>} />
      <Route path={ROUTES.deploymentDetail} element={<div>deployment detail</div>} />
    </Routes>
  );
}

describe("HoldoutPage", () => {
  it("lists every holdout evaluation returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.holdout] });
    expect(
      await screen.findByRole("link", { name: HOLDOUT_DEMO_SUMMARY.name }),
    ).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.holdout] });

    const link = await screen.findByRole("link", { name: HOLDOUT_DEMO_SUMMARY.name });
    await user.click(link);
    expect(await screen.findByText("holdout detail")).toBeInTheDocument();
  });

  it("deploys a holdout source and navigates to the new deployment", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.holdout] });

    await user.click(await screen.findByTestId(`deploy-holdout-${HOLDOUT_DEMO_SUMMARY.name}`));
    expect(await screen.findByText("deployment detail")).toBeInTheDocument();
  });
});
