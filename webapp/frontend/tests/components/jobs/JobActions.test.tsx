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

  it("renders the view-run link only for completed jobs with an experiment_id", () => {
    renderWithProviders(<JobActions job={JOB_COMPLETED} />);
    expect(screen.getByTestId("job-view-run-link")).toHaveAttribute(
      "href",
      `/runs/${JOB_COMPLETED.experiment_id ?? ""}`,
    );
  });

  it("does not render the view-run link for failed jobs", () => {
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
