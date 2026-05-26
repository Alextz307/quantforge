import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { HpoDetailPage } from "@/pages/HpoDetailPage";
import { API_PATHS, toMswPath } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import {
  HPO_DEMO_DETAIL,
  HPO_DEMO_IMPORTANCE,
  HPO_DEMO_IMPORTANCE_EMPTY,
  HPO_DEMO_SUMMARY,
  HPO_DEMO_TRIALS,
} from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.hpoDetail} element={<HpoDetailPage />} />
    </Routes>
  );
}

describe("HpoDetailPage", () => {
  it("renders the trial table, convergence chart, importance chart and best-config card", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });

    expect(await screen.findByTestId("trial-table")).toBeInTheDocument();
    expect(await screen.findByTestId("hpo-convergence")).toBeInTheDocument();
    expect(await screen.findByTestId("hpo-importance")).toBeInTheDocument();
    expect(await screen.findByTestId("best-config-json")).toBeInTheDocument();
  });

  it("renders one row per trial and highlights the best trial number", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });

    for (const t of HPO_DEMO_TRIALS) {
      expect(await screen.findByTestId(`trial-row-${String(t.number)}`)).toBeInTheDocument();
    }
    const bestRow = await screen.findByTestId(
      `trial-row-${String(HPO_DEMO_DETAIL.best_trial_number)}`,
    );
    expect(bestRow.className).toMatch(/bg-primary/);
  });

  it("renders the importance value count from the API response", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });

    const importance = await screen.findByTestId("hpo-importance");
    expect(importance.getAttribute("data-param-count")).toBe(
      String(Object.keys(HPO_DEMO_IMPORTANCE.importance).length),
    );
  });

  it("renders the empty-state message when importance has no completed trials yet", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.hpoParamImportance), () =>
        HttpResponse.json(HPO_DEMO_IMPORTANCE_EMPTY),
      ),
    );
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });

    const empty = await screen.findByTestId("hpo-importance-empty");
    expect(empty).toHaveTextContent(HPO_DEMO_IMPORTANCE_EMPTY.message ?? "");
  });

  it("hides the connection indicator when the study is not live", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });

    await screen.findByTestId("hpo-convergence");
    expect(screen.queryByTestId("connection-indicator")).toBeNull();
  });

  it("shows the connection indicator when the study has a live job id", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.hpoStudy), () =>
        HttpResponse.json({ ...HPO_DEMO_DETAIL, live_job_id: "job-uuid-tune" }),
      ),
    );
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });

    await waitFor(() => {
      expect(screen.getByTestId("connection-indicator")).toBeInTheDocument();
    });
  });

  it("shows the 'Run holdout eval' CTA when best_config reserves a holdout region", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });
    const cta = await screen.findByTestId("hpo-detail-holdout-cta");
    expect(cta).toHaveAttribute(
      "href",
      `${ROUTES.configureHoldout}?source_kind=hpo&source_id=${HPO_DEMO_SUMMARY.wire_id}`,
    );
  });

  it("hides the holdout CTA when best_config does not reserve a holdout region", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.hpoStudy), () =>
        HttpResponse.json({ ...HPO_DEMO_DETAIL, best_config_reserves_holdout: false }),
      ),
    );
    renderWithProviders(<Tree />, {
      initialEntries: [`/hpo/${HPO_DEMO_SUMMARY.wire_id}`],
    });
    await screen.findByTestId("hpo-convergence");
    expect(screen.queryByTestId("hpo-detail-holdout-cta")).not.toBeInTheDocument();
  });
});
