import { screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { EquityOverlayChart } from "@/components/charts/EquityOverlayChart";
import { API_PATHS, toMswPath } from "@/api/paths";
import { RUN_IVV_VOO, RUN_PAIRS_FOLDS, RUN_SPY, RUN_SPY_FOLDS } from "../../msw/handlers";
import { server } from "../../msw/server";
import { renderWithProviders } from "../../util/render";

const TWO_SPECS = [
  { label: RUN_SPY.strategy, experimentId: RUN_SPY.experiment_id },
  { label: RUN_IVV_VOO.strategy, experimentId: RUN_IVV_VOO.experiment_id },
];

describe("EquityOverlayChart", () => {
  it("renders one trace per spec when every linked run resolves", async () => {
    renderWithProviders(<EquityOverlayChart specs={TWO_SPECS} />);

    const overlay = await screen.findByTestId("equity-overlay");
    expect(overlay).toHaveAttribute("data-trace-count", String(TWO_SPECS.length));

    const plot = await screen.findByTestId("plotly-plot");
    const names = plot.getAttribute("data-trace-names")?.split(",") ?? [];
    expect(names).toContain(RUN_SPY.strategy);
    expect(names).toContain(RUN_IVV_VOO.strategy);

    expect(RUN_SPY_FOLDS.length).toBeGreaterThan(0);
    expect(RUN_PAIRS_FOLDS.length).toBeGreaterThan(0);
  });

  it("falls back to a 'failed labels' note when one of the linked runs errors", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.runFolds), ({ params }) => {
        if (params.experiment_id === RUN_IVV_VOO.experiment_id)
          return new HttpResponse(null, { status: 500 });
        return HttpResponse.json(RUN_SPY_FOLDS);
      }),
    );

    renderWithProviders(<EquityOverlayChart specs={TWO_SPECS} />);

    const overlay = await screen.findByTestId("equity-overlay");
    expect(overlay).toHaveAttribute("data-trace-count", "1");

    const failed = await screen.findByTestId("equity-overlay-failed-labels");
    expect(failed).toHaveTextContent(RUN_IVV_VOO.strategy);
  });
});
