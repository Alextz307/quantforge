import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ConfigureComparePage } from "@/pages/ConfigureComparePage";
import { API_PATHS } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { RUN_IVV_VOO, RUN_SPY } from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

const COMPARE_OUT_NAME = "my_compare";

// Sibling of RUN_SPY sharing the same data_hash — represents a different
// strategy run against the same SPY bar series, which is the canonical
// reuse-mode compare workflow.
const RUN_SPY_VT = {
  ...RUN_SPY,
  experiment_id: "exp_spy_vt",
  name: "spy_daily_5y_vt",
  strategy: "VolatilityTargeting",
};

function seedMatchingHashPair() {
  server.use(
    http.get(API_PATHS.runs, () =>
      HttpResponse.json({
        items: [RUN_SPY, RUN_SPY_VT],
        total: 2,
        limit: 200,
        offset: 0,
      }),
    ),
  );
}

describe("ConfigureComparePage", () => {
  it("submits a valid compare payload and navigates to the new job's detail page", async () => {
    seedMatchingHashPair();
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configureCompare} element={<ConfigureComparePage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configureCompare] },
    );

    // Wait for the runs picker to settle.
    await screen.findByText(RUN_SPY.name);
    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));
    await user.click(screen.getByLabelText(`Select ${RUN_SPY_VT.name}`));

    await user.type(screen.getByLabelText(/Output name/i), COMPARE_OUT_NAME);
    await user.click(screen.getByRole("button", { name: /Launch comparison/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("blocks submit and shows an inline error when fewer than 2 runs are selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ConfigureComparePage />);

    await screen.findByText(RUN_SPY.name);
    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));
    await user.type(screen.getByLabelText(/Output name/i), COMPARE_OUT_NAME);

    await user.click(screen.getByRole("button", { name: /Launch comparison/i }));

    expect(await screen.findByText(/Pick at least 2 runs/i)).toBeInTheDocument();
  });

  it("disables runs whose data_hash differs from the first selection", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ConfigureComparePage />);

    await screen.findByText(RUN_SPY.name);
    // Picker is unrestricted before any selection.
    expect(screen.getByLabelText(`Select ${RUN_IVV_VOO.name}`)).not.toBeDisabled();

    // Selecting RUN_SPY locks the picker to its bar series.
    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));

    // RUN_IVV_VOO has a different data_hash → disabled + a lock notice appears.
    expect(screen.getByLabelText(`Select ${RUN_IVV_VOO.name}`)).toBeDisabled();
    expect(screen.getByTestId("compare-data-hash-lock-notice")).toBeInTheDocument();

    // Deselecting clears the lock.
    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));
    expect(screen.queryByTestId("compare-data-hash-lock-notice")).not.toBeInTheDocument();
    expect(screen.getByLabelText(`Select ${RUN_IVV_VOO.name}`)).not.toBeDisabled();
  });

  it("surfaces backend 422 errors inline (e.g. ghost run id)", async () => {
    seedMatchingHashPair();
    server.use(
      http.post(API_PATHS.jobs, () =>
        HttpResponse.json(
          {
            detail: [
              {
                loc: ["compare_payload", "run_ids", "1"],
                msg: "run not found: ghost_run",
                type: "value_error",
              },
            ],
          },
          { status: 422 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigureComparePage />);

    await screen.findByText(RUN_SPY.name);
    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));
    await user.click(screen.getByLabelText(`Select ${RUN_SPY_VT.name}`));
    await user.type(screen.getByLabelText(/Output name/i), COMPARE_OUT_NAME);

    await user.click(screen.getByRole("button", { name: /Launch comparison/i }));

    expect(await screen.findByText(/run not found: ghost_run/i)).toBeInTheDocument();
  });

  it("caps selection at 8 runs and disables additional checkboxes", async () => {
    // Override /api/runs to return 9 runs so we can hit the cap.
    server.use(
      http.get(API_PATHS.runs, () =>
        HttpResponse.json({
          items: Array.from({ length: 9 }, (_, i) => ({
            ...RUN_SPY,
            experiment_id: `exp_run_${String(i)}`,
            name: `run_${String(i)}`,
          })),
          total: 9,
          limit: 200,
          offset: 0,
        }),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigureComparePage />);

    await screen.findByText("run_0");
    // Select the first 8 runs.
    for (let i = 0; i < 8; i += 1) {
      await user.click(screen.getByLabelText(`Select run_${String(i)}`));
    }
    const ninth = screen.getByLabelText("Select run_8");
    await waitFor(() => {
      expect(ninth).toBeDisabled();
    });
  });

  it("renders an empty-state message when no runs are available", async () => {
    server.use(
      http.get(API_PATHS.runs, () =>
        HttpResponse.json({ items: [], total: 0, limit: 200, offset: 0 }),
      ),
    );
    renderWithProviders(<ConfigureComparePage />);

    expect(await screen.findByText(/No completed runs yet/i)).toBeInTheDocument();
    // Sanity: the submit button is still rendered.
    expect(
      within(document.body).getByRole("button", { name: /Launch comparison/i }),
    ).toBeInTheDocument();
  });
});
