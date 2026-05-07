import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { JobStatusPill } from "@/components/jobs/JobStatusPill";
import type { JobStatus } from "@/api/jobs";

const STATUSES: JobStatus[] = ["queued", "running", "completed", "failed", "cancelled"];

describe("JobStatusPill", () => {
  it.each(STATUSES)("renders the status label and data attribute for %s", (status) => {
    render(<JobStatusPill status={status} />);
    const pill = screen.getByTestId("job-status-pill");
    expect(pill.dataset.status).toBe(status);
    expect(pill.textContent).toContain(status);
  });
});
