import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ConfigureHoldoutPage } from "@/pages/ConfigureHoldoutPage";
import { API_PATHS } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { HPO_DEMO_SUMMARY, RUN_IVV_VOO, RUN_SPY } from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

describe("ConfigureHoldoutPage", () => {
  it("submits a holdout payload from a run source and navigates to the job", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configureHoldout} element={<ConfigureHoldoutPage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configureHoldout] },
    );

    await screen.findByText(RUN_SPY.name);
    expect(screen.queryByText(RUN_IVV_VOO.name)).not.toBeInTheDocument();

    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));
    await user.click(screen.getByRole("button", { name: /Launch holdout/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("blocks submit when no source is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ConfigureHoldoutPage />);

    await screen.findByText(RUN_SPY.name);
    await user.click(screen.getByRole("button", { name: /Launch holdout/i }));

    expect(await screen.findByText(/Pick a source/i)).toBeInTheDocument();
  });

  it("filters HPO studies to those whose best_config reserves holdout", async () => {
    server.use(
      http.get(API_PATHS.hpoStudies, () =>
        HttpResponse.json([
          {
            ...HPO_DEMO_SUMMARY,
            name: "ready_for_holdout",
            has_best_config: true,
            best_config_reserves_holdout: true,
          },
          {
            ...HPO_DEMO_SUMMARY,
            name: "no_best_config",
            has_best_config: false,
            best_config_reserves_holdout: false,
          },
          {
            ...HPO_DEMO_SUMMARY,
            name: "no_holdout_reservation",
            has_best_config: true,
            best_config_reserves_holdout: false,
          },
        ]),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigureHoldoutPage />);

    await user.click(screen.getByLabelText(/From HPO best config/i));

    await screen.findByText("ready_for_holdout");
    expect(screen.queryByText("no_best_config")).not.toBeInTheDocument();
    expect(screen.queryByText("no_holdout_reservation")).not.toBeInTheDocument();
  });

  it("prefills the source kind + id from query params", async () => {
    renderWithProviders(<ConfigureHoldoutPage />, {
      initialEntries: [
        `${ROUTES.configureHoldout}?source_kind=hpo&source_id=${HPO_DEMO_SUMMARY.wire_id}`,
      ],
    });

    await screen.findByTestId("holdout-hpo-picker");
    const radio = await screen.findByLabelText(`Select ${HPO_DEMO_SUMMARY.name}`);
    expect(radio).toBeChecked();
  });

  it("surfaces backend 422 errors inline (e.g. missing holdout boundary)", async () => {
    server.use(
      http.post(API_PATHS.jobs, () =>
        HttpResponse.json(
          {
            detail: [
              {
                loc: ["holdout_payload", "source_id"],
                msg: "run exp_spy has no holdout boundary",
                type: "value_error",
              },
            ],
          },
          { status: 422 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigureHoldoutPage />);

    await screen.findByText(RUN_SPY.name);
    await user.click(screen.getByLabelText(`Select ${RUN_SPY.name}`));
    await user.click(screen.getByRole("button", { name: /Launch holdout/i }));

    expect(await screen.findByText(/no holdout boundary/i)).toBeInTheDocument();
  });
});
