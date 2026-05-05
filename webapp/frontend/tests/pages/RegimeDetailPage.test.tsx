import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RegimeDetailPage } from "@/pages/RegimeDetailPage";
import { ROUTES } from "@/lib/routes";
import { REGIME_DEMO_DETAIL, REGIME_DEMO_SUMMARY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.regimeDetail} element={<RegimeDetailPage />} />
    </Routes>
  );
}

describe("RegimeDetailPage", () => {
  it("renders identity, per-regime stats, and the timeline + heatmap charts", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/regime/${REGIME_DEMO_SUMMARY.name}`],
    });

    expect(await screen.findByText(REGIME_DEMO_DETAIL.detector_name)).toBeInTheDocument();
    expect(await screen.findByTestId("per-regime-stats-table")).toBeInTheDocument();
    expect(await screen.findByTestId("regime-timeline")).toBeInTheDocument();
    expect(await screen.findByTestId("regime-heatmap")).toBeInTheDocument();
  });

  it("links each plot file with download attribute", async () => {
    renderWithProviders(<Tree />, {
      initialEntries: [`/regime/${REGIME_DEMO_SUMMARY.name}`],
    });

    const link = await screen.findByRole("link", { name: "timeline.png" });
    expect(link).toHaveAttribute("download", "timeline.png");
  });
});
