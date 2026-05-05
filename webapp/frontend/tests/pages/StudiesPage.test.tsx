import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { StudiesPage } from "@/pages/StudiesPage";
import { ROUTES } from "@/lib/routes";
import { STUDY_DEMO_SUMMARY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.studies} element={<StudiesPage />} />
      <Route path={ROUTES.studyDetail} element={<div>study detail</div>} />
    </Routes>
  );
}

describe("StudiesPage", () => {
  it("lists every study returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.studies] });
    expect(await screen.findByRole("link", { name: STUDY_DEMO_SUMMARY.name })).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.studies] });

    const link = await screen.findByRole("link", { name: STUDY_DEMO_SUMMARY.name });
    await user.click(link);
    expect(await screen.findByText("study detail")).toBeInTheDocument();
  });
});
