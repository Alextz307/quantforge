import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { LegStatusGrid } from "@/components/studies/LegStatusGrid";
import { STUDY_DEMO_DETAIL, RUN_SPY } from "../../msw/handlers";
import { ROUTER_FUTURE_FLAGS } from "../../util/router";

const COMPLETED_LEG = STUDY_DEMO_DETAIL.legs.find((l) => l.is_complete);

function renderGrid(legs: typeof STUDY_DEMO_DETAIL.legs) {
  render(
    <MemoryRouter future={ROUTER_FUTURE_FLAGS}>
      <LegStatusGrid legs={legs} />
    </MemoryRouter>,
  );
}

describe("LegStatusGrid", () => {
  it("renders an empty state when no legs are provided", () => {
    renderGrid([]);
    expect(screen.getByText(/no legs in this study/i)).toBeInTheDocument();
  });

  it("renders one cell per leg with the matching status", () => {
    renderGrid(STUDY_DEMO_DETAIL.legs);
    for (const leg of STUDY_DEMO_DETAIL.legs) {
      expect(screen.getByTestId(`leg-cell-${leg.leg_id}`)).toBeInTheDocument();
    }
  });

  it("links a complete leg back to its underlying run", () => {
    renderGrid(STUDY_DEMO_DETAIL.legs);
    if (!COMPLETED_LEG) throw new Error("expected at least one complete leg in fixtures");
    const cell = screen.getByTestId(`leg-cell-${COMPLETED_LEG.leg_id}`);
    expect(cell).toHaveAttribute("href", `/runs/${RUN_SPY.experiment_id}`);
  });
});
