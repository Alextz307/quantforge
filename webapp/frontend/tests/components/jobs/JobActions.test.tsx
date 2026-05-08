import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { JobActions } from "@/components/jobs/JobActions";
import { API_PATHS, toMswPath } from "@/api/paths";
import type { JobRow } from "@/api/jobs";
import { JOB_COMPLETED, JOB_FAILED, JOB_RUNNING } from "../../msw/handlers";
import { server } from "../../msw/server";
import { renderWithProviders } from "../../util/render";

describe("JobActions", () => {
  it("disables cancel for terminal jobs", () => {
    renderWithProviders(<JobActions job={JOB_COMPLETED} />);
    expect(screen.getByRole("button", { name: /Cancel/i })).toBeDisabled();
  });

  it("renders the view-run link for completed run jobs", () => {
    renderWithProviders(<JobActions job={JOB_COMPLETED} />);
    const link = screen.getByTestId("job-view-run-link");
    expect(link).toHaveAttribute("href", `/runs/${JOB_COMPLETED.experiment_id ?? ""}`);
    expect(link).toHaveTextContent(/View run/);
  });

  it("renders the view-study link for completed tune jobs", () => {
    const tuneJob: JobRow = {
      ...JOB_COMPLETED,
      kind: "tune",
      experiment_id: "demo_study",
    };
    renderWithProviders(<JobActions job={tuneJob} />);
    const link = screen.getByTestId("job-view-run-link");
    expect(link).toHaveAttribute("href", "/hpo/demo_study");
    expect(link).toHaveTextContent(/View study/);
  });

  it("does not render the artifact link for failed jobs", () => {
    renderWithProviders(<JobActions job={JOB_FAILED} />);
    expect(screen.queryByTestId("job-view-run-link")).not.toBeInTheDocument();
  });

  it("posts DELETE on cancel-confirm and surfaces errors", async () => {
    let deleted = false;
    server.use(
      http.delete(toMswPath(API_PATHS.job), () => {
        deleted = true;
        return HttpResponse.json({ ...JOB_RUNNING, status: "cancelled" } satisfies JobRow);
      }),
    );
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const user = userEvent.setup();

    renderWithProviders(<JobActions job={JOB_RUNNING} />);
    await user.click(screen.getByRole("button", { name: /Cancel/i }));

    expect(confirmSpy).toHaveBeenCalled();
    await vi.waitFor(() => {
      expect(deleted).toBe(true);
    });
    confirmSpy.mockRestore();
  });
});
