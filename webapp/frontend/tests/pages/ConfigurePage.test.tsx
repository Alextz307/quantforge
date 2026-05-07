import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ConfigurePage } from "@/pages/ConfigurePage";
import { API_PATHS } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { PUBLIC_SETTINGS_DISABLED } from "../msw/handlers";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

async function fillBaseFields(user: ReturnType<typeof userEvent.setup>): Promise<void> {
  await user.type(screen.getByLabelText(/Run name/i), "demo");
  await user.clear(screen.getByLabelText(/Tickers/i));
  await user.type(screen.getByLabelText(/Tickers/i), "SPY");
  await user.clear(screen.getByLabelText(/Start/i));
  await user.type(screen.getByLabelText(/Start/i), "2020-01-01");
  await user.clear(screen.getByLabelText(/End/i));
  await user.type(screen.getByLabelText(/End/i), "2024-12-31");
  await user.selectOptions(screen.getByLabelText(/Strategy$/i), "AdaptiveBollinger");
}

describe("ConfigurePage", () => {
  it("renders the disabled state when jobs_enabled is false", async () => {
    server.use(
      http.get(API_PATHS.publicSettings, () => HttpResponse.json(PUBLIC_SETTINGS_DISABLED)),
    );

    renderWithProviders(<ConfigurePage />);

    expect(await screen.findByText(/Job execution is disabled/i)).toBeInTheDocument();
  });

  it("submits a valid payload and navigates to the new job's detail page", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configure} element={<ConfigurePage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configure] },
    );

    // Wait for /api/strategies + /api/settings/public to settle.
    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillBaseFields(user);
    await user.click(screen.getByRole("button", { name: /Launch run/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("renders backend validation errors inline when /configs/validate fails", async () => {
    server.use(
      http.post(API_PATHS.configValidate, () =>
        HttpResponse.json({
          valid: false,
          errors: [{ loc: ["data", "tickers"], msg: "must be non-empty", type: "value_error" }],
        }),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigurePage />);

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillBaseFields(user);
    await user.click(screen.getByRole("button", { name: /Launch run/i }));

    expect(await screen.findByText(/must be non-empty/i)).toBeInTheDocument();
    expect(screen.getByText(/data\.tickers/i)).toBeInTheDocument();
  });
});
