import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ComparisonsPage } from "@/pages/ComparisonsPage";
import { ROUTES } from "@/lib/routes";
import { COMPARISON_DEMO_SUMMARY, RUN_IVV_VOO } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.comparisons} element={<ComparisonsPage />} />
      <Route path={ROUTES.comparisonDetail} element={<div>comparison detail</div>} />
    </Routes>
  );
}

describe("ComparisonsPage", () => {
  it("lists every comparison returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.comparisons] });
    expect(
      await screen.findByRole("link", { name: COMPARISON_DEMO_SUMMARY.name }),
    ).toBeInTheDocument();
  });

  it("filters by selected strategy", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.comparisons] });

    await screen.findByRole("link", { name: COMPARISON_DEMO_SUMMARY.name });
    const table = screen.getByTestId("comparisons-table");
    expect(within(table).getByText(COMPARISON_DEMO_SUMMARY.name)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/strategy/i), RUN_IVV_VOO.strategy);
    expect(within(table).getByText(COMPARISON_DEMO_SUMMARY.name)).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.comparisons] });

    const link = await screen.findByRole("link", { name: COMPARISON_DEMO_SUMMARY.name });
    await user.click(link);
    expect(await screen.findByText("comparison detail")).toBeInTheDocument();
  });
});
