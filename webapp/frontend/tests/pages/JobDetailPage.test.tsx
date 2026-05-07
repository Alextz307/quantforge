import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { JobDetailPage } from "@/pages/JobDetailPage";
import { ROUTES } from "@/lib/routes";
import { JOB_COMPLETED } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

describe("JobDetailPage", () => {
  it("renders the job header with status pill and exit code for completed jobs", async () => {
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.jobDetail} element={<JobDetailPage />} />
      </Routes>,
      { initialEntries: [`/jobs/${JOB_COMPLETED.id}`] },
    );

    expect(await screen.findByText(JOB_COMPLETED.id)).toBeInTheDocument();
    expect(await screen.findByTestId("job-status-pill")).toHaveAttribute(
      "data-status",
      "completed",
    );
    expect(screen.getByText(/exit: 0/i)).toBeInTheDocument();
    expect(screen.getByTestId("job-view-run-link")).toBeInTheDocument();
  });
});
