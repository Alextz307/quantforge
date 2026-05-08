import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ConfigureTunePage } from "@/pages/ConfigureTunePage";
import { API_PATHS } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

const STUDY_NAME = "tune_demo";

async function fillExperimentBlock(user: ReturnType<typeof userEvent.setup>): Promise<void> {
  await user.type(screen.getByLabelText(/Run name/i), "demo");
  await user.clear(screen.getByLabelText(/Tickers/i));
  await user.type(screen.getByLabelText(/Tickers/i), "SPY");
  await user.clear(screen.getByLabelText(/Start/i));
  await user.type(screen.getByLabelText(/Start/i), "2020-01-01");
  await user.clear(screen.getByLabelText(/End/i));
  await user.type(screen.getByLabelText(/End/i), "2024-12-31");
  await user.selectOptions(screen.getByLabelText(/Strategy$/i), "AdaptiveBollinger");
}

describe("ConfigureTunePage", () => {
  it("submits a JobSubmission with kind=tune and the HPO payload", async () => {
    const submissions: Array<Record<string, unknown>> = [];
    server.use(
      http.post(API_PATHS.jobs, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        submissions.push(body);
        return HttpResponse.json(
          {
            id: "job-uuid-tune",
            user_id: 1,
            kind: "tune",
            status: "queued",
            started_at: null,
            finished_at: null,
            exit_code: null,
            experiment_id: STUDY_NAME,
            log_path: "/tmp/jobs/job-uuid-tune.log",
            pid: null,
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configureTune} element={<ConfigureTunePage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configureTune] },
    );

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillExperimentBlock(user);
    await user.type(screen.getByLabelText(/Study name/i), STUDY_NAME);
    await user.click(screen.getByRole("button", { name: /Launch tune/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
    expect(submissions).toHaveLength(1);
    const submission = submissions[0];
    if (!submission) throw new Error("submission was not captured");
    expect(submission.kind).toBe("tune");
    expect(submission.config_payload).toBeTruthy();
    const hpo = submission.hpo_payload as Record<string, unknown>;
    expect(hpo.study_name).toBe(STUDY_NAME);
    expect(hpo.sampler).toBe("tpe");
    expect(hpo.objective).toBe("sharpe");
    expect(hpo.n_trials).toBe(50);
  });

  it("surfaces inline errors when /configs/validate?kind=hpo fails", async () => {
    server.use(
      http.post(API_PATHS.configValidate, async ({ request }) => {
        const body = (await request.json()) as { kind: string };
        if (body.kind === "hpo") {
          return HttpResponse.json({
            valid: false,
            errors: [
              { loc: ["study_name"], msg: "must not contain path separators", type: "value_error" },
            ],
          });
        }
        return HttpResponse.json({ valid: true, errors: [] });
      }),
    );

    const user = userEvent.setup();
    renderWithProviders(<ConfigureTunePage />);

    await screen.findByLabelText(/Run name/i);
    await waitFor(() => {
      expect(screen.getByLabelText(/Strategy$/i)).not.toBeDisabled();
    });

    await fillExperimentBlock(user);
    await user.type(screen.getByLabelText(/Study name/i), "anything_goes");
    await user.click(screen.getByRole("button", { name: /Launch tune/i }));

    expect(await screen.findByText(/must not contain path separators/i)).toBeInTheDocument();
    expect(screen.getByText(/hpo_payload\.study_name/i)).toBeInTheDocument();
  });
});
