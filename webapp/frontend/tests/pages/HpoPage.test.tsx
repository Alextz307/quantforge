import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { HpoPage } from "@/pages/HpoPage";
import { ROUTES } from "@/lib/routes";
import { HPO_DEMO_SUMMARY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.hpo} element={<HpoPage />} />
      <Route path={ROUTES.hpoDetail} element={<div>hpo detail</div>} />
    </Routes>
  );
}

describe("HpoPage", () => {
  it("lists every HPO study returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.hpo] });
    expect(
      await screen.findByRole("link", { name: HPO_DEMO_SUMMARY.name }),
    ).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.hpo] });

    const link = await screen.findByRole("link", { name: HPO_DEMO_SUMMARY.name });
    await user.click(link);
    expect(await screen.findByText("hpo detail")).toBeInTheDocument();
  });
});
