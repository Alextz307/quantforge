import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { HpoDetailPage } from "@/pages/HpoDetailPage";
import { ROUTES } from "@/lib/routes";
import { HPO_DEMO_DETAIL, HPO_DEMO_SUMMARY, HPO_DEMO_TRIALS } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.hpoDetail} element={<HpoDetailPage />} />
    </Routes>
  );
}

describe("HpoDetailPage", () => {
  it("renders the trial table, convergence chart and best-config card", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.name}`],
    });

    expect(await screen.findByTestId("trial-table")).toBeInTheDocument();
    expect(await screen.findByTestId("hpo-convergence")).toBeInTheDocument();
    expect(await screen.findByTestId("best-config-json")).toBeInTheDocument();
  });

  it("renders one row per trial and highlights the best trial number", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.name}`],
    });

    for (const t of HPO_DEMO_TRIALS) {
      expect(
        await screen.findByTestId(`trial-row-${String(t.number)}`),
      ).toBeInTheDocument();
    }
    const bestRow = await screen.findByTestId(
      `trial-row-${String(HPO_DEMO_DETAIL.best_trial_number)}`,
    );
    expect(bestRow.className).toMatch(/bg-primary/);
  });
});
