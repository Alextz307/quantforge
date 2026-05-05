import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { StudyDetailPage } from "@/pages/StudyDetailPage";
import { ROUTES } from "@/lib/routes";
import {
  STUDY_CONSOLIDATED_DEMO,
  STUDY_DEMO_DETAIL,
  STUDY_DEMO_SUMMARY,
} from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.studyDetail} element={<StudyDetailPage />} />
    </Routes>
  );
}

describe("StudyDetailPage", () => {
  it("renders the leg-status grid + consolidated report panel when both endpoints succeed", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/studies/${STUDY_DEMO_SUMMARY.name}`],
    });

    expect(await screen.findByTestId("leg-status-grid")).toBeInTheDocument();
    expect(await screen.findByTestId("consolidated-report-panel")).toBeInTheDocument();
    expect(await screen.findByText(STUDY_CONSOLIDATED_DEMO.publish_label)).toBeInTheDocument();
  });

  it("renders one cell per (strategy, universe) leg", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/studies/${STUDY_DEMO_SUMMARY.name}`],
    });

    for (const leg of STUDY_DEMO_DETAIL.legs) {
      expect(await screen.findByTestId(`leg-cell-${leg.leg_id}`)).toBeInTheDocument();
    }
  });
});
