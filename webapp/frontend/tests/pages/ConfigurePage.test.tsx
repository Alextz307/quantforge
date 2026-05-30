import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ConfigurePage } from "@/pages/ConfigurePage";
import type { JobRow, JobSubmission } from "@/api/jobs";
import { API_PATHS } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { JOB_COMPLETED } from "../msw/handlers";
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
  it("submits a valid payload and navigates to the new job's detail page", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configure} element={<ConfigurePage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configure] },
    );

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillBaseFields(user);
    await user.click(screen.getByRole("button", { name: /Launch run/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("sends feature_importance=true when the toggle is checked", async () => {
    let capturedBody: JobSubmission | null = null;
    server.use(
      http.post(API_PATHS.jobs, async ({ request }) => {
        capturedBody = (await request.json()) as JobSubmission;
        return HttpResponse.json(JOB_COMPLETED satisfies JobRow);
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configure} element={<ConfigurePage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configure] },
    );

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillBaseFields(user);
    await user.click(screen.getByTestId("feature-importance-toggle"));
    await user.click(screen.getByRole("button", { name: /Launch run/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
    expect(capturedBody).not.toBeNull();
    expect((capturedBody as unknown as JobSubmission).feature_importance).toBe(true);
  });

  it("blocks submit and shows inline error when a required strategy param is missing", async () => {
    const validateCalls = { count: 0 };
    server.use(
      http.get(API_PATHS.strategies, () =>
        HttpResponse.json([
          {
            name: "AdaptiveBollinger",
            qualname: "src.strategies.adaptive.AdaptiveBollinger",
          },
        ]),
      ),
      http.get("*/api/strategies/AdaptiveBollinger/schema", () =>
        HttpResponse.json({
          name: "AdaptiveBollinger",
          qualname: "src.strategies.adaptive.AdaptiveBollinger",
          params: [
            {
              name: "feature_columns",
              kind: "complex",
              default: null,
              required: true,
              nullable: false,
              choices: null,
            },
          ],
        }),
      ),
      http.post(API_PATHS.configValidate, () => {
        validateCalls.count += 1;
        return HttpResponse.json({ valid: true, errors: [] });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigurePage />);

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillBaseFields(user);
    await user.click(screen.getByRole("button", { name: /Launch run/i }));

    expect(await screen.findByText(/strategy\.params\.feature_columns/i)).toBeInTheDocument();
    expect(screen.getAllByText(/field required/i).length).toBeGreaterThan(0);
    expect(validateCalls.count).toBe(0);
  });

  it("pre-fills strategy params from canonical_params when the strategy is picked", async () => {
    server.use(
      http.get(API_PATHS.strategies, () =>
        HttpResponse.json([
          { name: "AdaptiveBollinger", qualname: "src.strategies.adaptive.AdaptiveBollinger" },
        ]),
      ),
      http.get("*/api/strategies/AdaptiveBollinger/schema", () =>
        HttpResponse.json({
          name: "AdaptiveBollinger",
          qualname: "src.strategies.adaptive.AdaptiveBollinger",
          params: [
            {
              name: "window",
              kind: "int",
              default: 20,
              required: false,
              nullable: false,
              choices: null,
            },
            {
              name: "k",
              kind: "float",
              default: 2.0,
              required: false,
              nullable: false,
              choices: null,
            },
          ],
          canonical_params: { window: 25, k: 1.75 },
        }),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigurePage />);

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });
    await user.selectOptions(screen.getByLabelText(/Strategy$/i), "AdaptiveBollinger");

    await waitFor(() => {
      expect(screen.getByLabelText(/window/i)).toHaveValue(25);
    });
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
