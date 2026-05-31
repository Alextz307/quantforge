import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes, useLocation } from "react-router-dom";
import { HoldoutPage } from "@/pages/HoldoutPage";
import { ROUTES } from "@/lib/routes";
import { HOLDOUT_DEMO_SUMMARY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location-search">{location.search}</div>;
}

function Tree() {
  return (
    <Routes>
      <Route
        path={ROUTES.holdout}
        element={
          <>
            <HoldoutPage />
            <LocationProbe />
          </>
        }
      />
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

  it("persists the sort column in the URL so it survives navigation", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.holdout] });
    await screen.findByRole("link", { name: HOLDOUT_DEMO_SUMMARY.name });

    await user.click(screen.getByRole("button", { name: /Sharpe/i }));

    expect(screen.getByTestId("location-search")).toHaveTextContent("sort_by=sharpe_ratio");
  });

  it("deploys a holdout source and navigates to the new deployment", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.holdout] });

    await user.click(await screen.findByTestId(`deploy-holdout-${HOLDOUT_DEMO_SUMMARY.name}`));
    expect(await screen.findByText("deployment detail")).toBeInTheDocument();
  });

  it("keeps a URL source_kind filter selectable even when no loaded row matches", async () => {
    // ?source_kind=hpo with only run-sourced evals must show the filter active,
    // not silently fall back to "All".
    renderWithProviders(<Tree />, { initialEntries: [`${ROUTES.holdout}?source_kind=hpo`] });
    await screen.findByText(/No holdout evaluations match/i);

    const select = screen.getByLabelText<HTMLSelectElement>("Source kind");
    expect(select.value).toBe("hpo");
    expect(screen.getByRole("option", { name: "HPO trial" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: HOLDOUT_DEMO_SUMMARY.name })).not.toBeInTheDocument();
  });

  it("ignores an unparseable since param instead of crashing or hiding every row", async () => {
    renderWithProviders(<Tree />, { initialEntries: [`${ROUTES.holdout}?since=not-a-date`] });
    expect(
      await screen.findByRole("link", { name: HOLDOUT_DEMO_SUMMARY.name }),
    ).toBeInTheDocument();
  });
});
