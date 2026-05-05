import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RegimePage } from "@/pages/RegimePage";
import { ROUTES } from "@/lib/routes";
import { REGIME_DEMO_SUMMARY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.regime} element={<RegimePage />} />
      <Route path={ROUTES.regimeDetail} element={<div>regime detail</div>} />
    </Routes>
  );
}

describe("RegimePage", () => {
  it("lists every regime report returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.regime] });
    expect(
      await screen.findByRole("link", { name: REGIME_DEMO_SUMMARY.name }),
    ).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.regime] });

    const link = await screen.findByRole("link", { name: REGIME_DEMO_SUMMARY.name });
    await user.click(link);
    expect(await screen.findByText("regime detail")).toBeInTheDocument();
  });
});
