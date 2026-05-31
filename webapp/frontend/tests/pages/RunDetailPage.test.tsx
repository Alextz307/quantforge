import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RunDetailPage } from "@/pages/RunDetailPage";
import { API_PATHS, toMswPath } from "@/api/paths";
import type { JobRow } from "@/api/jobs";
import type { FeatureImportanceResponse } from "@/api/runs";
import { ROUTES, runDetailPath } from "@/lib/routes";
import { JOB_RUNNING, RUN_SPY, RUN_SPY_DETAIL, RUN_SPY_FOLDS } from "../msw/handlers";
import { server } from "../msw/server";
import { installMockWebSocket, MockWebSocket } from "../util/mockWebSocket";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.runs} element={<div>runs list</div>} />
      <Route path={ROUTES.runDetail} element={<RunDetailPage />} />
      <Route path={ROUTES.deploymentDetail} element={<div>deployment detail</div>} />
    </Routes>
  );
}

const IMPORTANCE_JOB_ID = "job-uuid-newly-submitted";

// A feature-consuming run that has no importance yet: the read endpoint reports
// it as computable so the UI offers the on-demand compute action.
const COMPUTABLE_EMPTY: FeatureImportanceResponse = {
  computable: true,
  entries: [],
  message: "Feature importance was not computed for this run.",
};

const COMPUTABLE_WITH_ENTRIES: FeatureImportanceResponse = {
  computable: true,
  entries: [{ feature: "rsi_14", importance: 0.4, std: 0.05, n_folds: 3, method: "permutation" }],
};

const DIVERGED_RUN_ID = "20260531_132236_VolatilityTargeting_13f2bd6_54ab2b68";

const COMPUTABLE_DIVERGED: FeatureImportanceResponse = {
  computable: true,
  entries: [],
  diverged_run_id: DIVERGED_RUN_ID,
  message: "Feature importance was not computed for this run.",
};

const RUNNING_IMPORTANCE_JOB: JobRow = {
  ...JOB_RUNNING,
  id: IMPORTANCE_JOB_ID,
  kind: "importance",
  status: "running",
  experiment_id: null,
};

function serveComputableEmpty(): void {
  server.use(
    http.get(toMswPath(API_PATHS.runFeatureImportance), () => HttpResponse.json(COMPUTABLE_EMPTY)),
    http.get(toMswPath(API_PATHS.job), () => HttpResponse.json(RUNNING_IMPORTANCE_JOB)),
  );
}

// The compute flow persists the in-flight job id in sessionStorage; clear it
// around every test (both describe blocks) so a started job never leaks into
// the next render.
beforeEach(() => {
  sessionStorage.clear();
});
afterEach(() => {
  sessionStorage.clear();
});

describe("RunDetailPage", () => {
  it("renders manifest fields, fold metrics, equity chart, and plot links", async () => {
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByText(RUN_SPY_DETAIL.name)).toBeInTheDocument();
    expect(screen.getByText(RUN_SPY_DETAIL.experiment_id)).toBeInTheDocument();
    expect(screen.getByText(RUN_SPY_DETAIL.strategy)).toBeInTheDocument();

    const chart = await screen.findByTestId("equity-chart");
    expect(chart).toHaveAttribute("data-trace-count", String(RUN_SPY_FOLDS.length));

    const plotLink = screen.getByRole("link", { name: "equity.png" });
    expect(plotLink).toHaveAttribute("href", `/api/runs/${RUN_SPY.experiment_id}/plots/equity.png`);
  });

  it("explains why a rule-based run has no feature importance", async () => {
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    const note = await screen.findByTestId("feature-importance-not-applicable");
    expect(note).toHaveTextContent(/rule-based strategy/i);
    expect(note).toHaveTextContent(RUN_SPY_DETAIL.strategy);
    expect(screen.queryByTestId("feature-importance-compute-button")).not.toBeInTheDocument();
  });

  it("shows the 'Run holdout eval' CTA when the manifest has a holdout boundary", async () => {
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });
    const cta = await screen.findByTestId("run-detail-holdout-cta");
    expect(cta).toHaveAttribute(
      "href",
      `${ROUTES.configureHoldout}?source_kind=run&source_id=${RUN_SPY.experiment_id}`,
    );
  });

  it("hides the holdout CTA when the manifest has no holdout boundary", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.run), ({ params }) =>
        HttpResponse.json({
          ...RUN_SPY_DETAIL,
          experiment_id: String(params.experiment_id),
          holdout_start: null,
        }),
      ),
    );
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });
    await screen.findByText(RUN_SPY_DETAIL.name);
    expect(screen.queryByTestId("run-detail-holdout-cta")).not.toBeInTheDocument();
  });

  it("warns when deploying a run that has no holdout evaluation", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.run), ({ params }) =>
        HttpResponse.json({
          ...RUN_SPY_DETAIL,
          experiment_id: String(params.experiment_id),
          holdout_start: null,
        }),
      ),
    );
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });
    expect(await screen.findByText("No holdout evaluation")).toBeInTheDocument();
  });

  it("deploys the run and navigates to the new deployment", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });
    await user.click(await screen.findByTestId("run-detail-deploy-cta"));
    expect(await screen.findByText("deployment detail")).toBeInTheDocument();
  });
});

describe("RunDetailPage feature-importance compute flow", () => {
  installMockWebSocket();

  it("offers the compute action for a feature-consuming run without importance", async () => {
    serveComputableEmpty();
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByTestId("feature-importance-compute-button")).toBeInTheDocument();
    expect(screen.getByTestId("feature-importance-compute")).toHaveTextContent(/re-trains/i);
  });

  it("links to the diverged run persistently across reloads", async () => {
    // A prior recompute diverged; the read endpoint reports the separate run,
    // so the link survives a page reload without any live job state.
    server.use(
      http.get(toMswPath(API_PATHS.runFeatureImportance), () =>
        HttpResponse.json(COMPUTABLE_DIVERGED),
      ),
    );
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByTestId("feature-importance-diverged-link")).toHaveAttribute(
      "href",
      runDetailPath(DIVERGED_RUN_ID),
    );
    expect(screen.getByTestId("feature-importance-compute-button")).toHaveTextContent(/recompute/i);
  });

  it("starts a job and shows the running watcher on click", async () => {
    serveComputableEmpty();
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    await user.click(await screen.findByTestId("feature-importance-compute-button"));

    expect(await screen.findByTestId("feature-importance-running")).toBeInTheDocument();
  });

  it("resumes the running watcher from storage after a reload", async () => {
    serveComputableEmpty();
    // Simulate a reload mid-computation: the in-flight job id is in
    // sessionStorage, so the watcher resumes instead of resetting to the button.
    sessionStorage.setItem(`importanceJob:${RUN_SPY.experiment_id}`, IMPORTANCE_JOB_ID);
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    expect(await screen.findByTestId("feature-importance-running")).toBeInTheDocument();
    expect(screen.queryByTestId("feature-importance-compute-button")).not.toBeInTheDocument();
  });

  it("links to the new run when the re-fit diverges", async () => {
    serveComputableEmpty();
    const divergedRunId = "exp_importance_refit";
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    await user.click(await screen.findByTestId("feature-importance-compute-button"));
    await screen.findByTestId("feature-importance-running");

    // A diverged re-run resolves to a new experiment_id and writes a divergence
    // pointer into this run. Mirror both so the diverged notice + link survive
    // the watcher handing back to the compute card on completion.
    server.use(
      http.get(toMswPath(API_PATHS.job), () =>
        HttpResponse.json({
          ...RUNNING_IMPORTANCE_JOB,
          status: "completed",
          exit_code: 0,
          experiment_id: divergedRunId,
        } satisfies JobRow),
      ),
      http.get(toMswPath(API_PATHS.runFeatureImportance), () =>
        HttpResponse.json({
          computable: true,
          entries: [],
          diverged_run_id: divergedRunId,
        } satisfies FeatureImportanceResponse),
      ),
    );

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    ws.triggerOpen();
    await waitFor(() => {
      ws.triggerMessage({
        type: "status",
        status: "completed",
        exit_code: 0,
        experiment_id: divergedRunId,
      });
      expect(screen.getByTestId("feature-importance-diverged")).toBeInTheDocument();
    });
    expect(await screen.findByTestId("feature-importance-diverged-link")).toHaveAttribute(
      "href",
      runDetailPath(divergedRunId),
    );
  });

  it("swaps in the chart when the re-fit reproduces and backfills in place", async () => {
    serveComputableEmpty();
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [runDetailPath(RUN_SPY.experiment_id)] });

    await user.click(await screen.findByTestId("feature-importance-compute-button"));
    await screen.findByTestId("feature-importance-running");

    // The backfill wrote importance into this run (refetch returns entries) and
    // left the job's experiment_id null (importance landed on the original run).
    server.use(
      http.get(toMswPath(API_PATHS.runFeatureImportance), () =>
        HttpResponse.json(COMPUTABLE_WITH_ENTRIES),
      ),
      http.get(toMswPath(API_PATHS.job), () =>
        HttpResponse.json({
          ...RUNNING_IMPORTANCE_JOB,
          status: "completed",
          exit_code: 0,
          experiment_id: null,
        } satisfies JobRow),
      ),
    );

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    ws.triggerOpen();
    await waitFor(() => {
      ws.triggerMessage({ type: "status", status: "completed", exit_code: 0, experiment_id: null });
      expect(screen.getByTestId("feature-importance")).toBeInTheDocument();
    });
  });
});
