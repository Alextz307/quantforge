import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RegimeTimeline } from "@/components/charts/RegimeTimeline";
import { REGIME_DEMO_DETAIL } from "../../msw/handlers";

describe("RegimeTimeline", () => {
  it("renders an empty state when no slices are provided", () => {
    render(<RegimeTimeline slices={[]} />);
    expect(screen.getByTestId("regime-timeline-empty")).toBeInTheDocument();
  });

  it("renders the chart wrapper with a slice count when slices exist", () => {
    render(<RegimeTimeline slices={REGIME_DEMO_DETAIL.slices} />);
    const wrapper = screen.getByTestId("regime-timeline");
    expect(wrapper.getAttribute("data-slice-count")).toBe(
      String(REGIME_DEMO_DETAIL.slices.length),
    );
  });
});
