import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { API_PATHS } from "@/api/paths";
import { HpoPage } from "@/pages/HpoPage";
import { ROUTES } from "@/lib/routes";
import { HPO_DEMO_SUMMARY } from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.hpo} element={<HpoPage />} />
      <Route path={ROUTES.hpoDetail} element={<div>hpo detail</div>} />
    </Routes>
  );
}

function rowOrder(): (string | null)[] {
  const table = screen.getByTestId("hpo-table");
  return within(table)
    .getAllByRole("link")
    .map((link) => link.textContent);
}

describe("HpoPage", () => {
  it("lists every HPO study returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.hpo] });
    expect(await screen.findByRole("link", { name: HPO_DEMO_SUMMARY.name })).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.hpo] });

    const link = await screen.findByRole("link", { name: HPO_DEMO_SUMMARY.name });
    await user.click(link);
    expect(await screen.findByText("hpo detail")).toBeInTheDocument();
  });

  it("sorts by best_value when the 'Best' header is clicked", async () => {
    // Three studies with monotonic best_value and reverse-monotonic created_at
    // so the default created-DESC order differs from best-DESC order.
    server.use(
      http.get(API_PATHS.hpoStudies, () =>
        HttpResponse.json([
          {
            ...HPO_DEMO_SUMMARY,
            name: "study_low",
            best_value: 0.1,
            created_at: "2026-05-01T00:00:00Z",
          },
          {
            ...HPO_DEMO_SUMMARY,
            name: "study_mid",
            best_value: 0.5,
            created_at: "2026-04-01T00:00:00Z",
          },
          {
            ...HPO_DEMO_SUMMARY,
            name: "study_high",
            best_value: 0.9,
            created_at: "2026-03-01T00:00:00Z",
          },
        ]),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.hpo] });

    // Default: created DESC -> low, mid, high.
    await screen.findByText("study_low");
    expect(rowOrder()).toEqual(["study_low", "study_mid", "study_high"]);

    // Click "Best" -> best DESC.
    await user.click(screen.getByRole("button", { name: /^Best/ }));
    expect(rowOrder()).toEqual(["study_high", "study_mid", "study_low"]);

    // Re-click "Best" -> best ASC.
    await user.click(screen.getByRole("button", { name: /^Best/ }));
    expect(rowOrder()).toEqual(["study_low", "study_mid", "study_high"]);
  });
});
