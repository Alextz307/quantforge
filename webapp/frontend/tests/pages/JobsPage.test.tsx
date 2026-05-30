import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { JobsPage } from "@/pages/JobsPage";
import { JOB_COMPLETED, JOB_RUNNING } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

describe("JobsPage", () => {
  it("renders the seeded jobs with their status pills and links", async () => {
    renderWithProviders(<JobsPage />);

    await screen.findByTestId("jobs-table");
    const pills = screen.getAllByTestId("job-status-pill");
    const statuses = pills.map((p) => p.dataset.status);
    expect(statuses).toEqual(expect.arrayContaining(["running", "completed", "failed"]));

    const runLink = screen.getByRole("link", { name: /view ->/i });
    expect(runLink).toHaveAttribute("href", `/runs/${JOB_COMPLETED.experiment_id ?? ""}`);
    expect(screen.getByTestId("jobs-launch-link")).toHaveAttribute("href", "/configure");
  });

  it("filters by status when the dropdown changes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<JobsPage />);
    await screen.findByTestId("jobs-table");

    await user.selectOptions(screen.getByLabelText(/Status/i), "running");

    const pills = screen.getAllByTestId("job-status-pill");
    expect(pills.length).toBe(1);
    expect(pills[0]?.dataset.status).toBe("running");
    expect(pills[0]?.textContent).toContain(JOB_RUNNING.status);
  });
});
