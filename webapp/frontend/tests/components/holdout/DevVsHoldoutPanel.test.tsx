import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { DevVsHoldoutPanel } from "@/components/holdout/DevVsHoldoutPanel";
import { HOLDOUT_DEMO_DETAIL, RUN_SPY } from "../../msw/handlers";
import { ROUTER_FUTURE_FLAGS } from "../../util/router";

describe("DevVsHoldoutPanel", () => {
  it("renders dev source identity, holdout metrics, and a link to the source run", () => {
    render(
      <MemoryRouter future={ROUTER_FUTURE_FLAGS}>
        <DevVsHoldoutPanel holdout={HOLDOUT_DEMO_DETAIL} />
      </MemoryRouter>,
    );

    expect(screen.getByText(HOLDOUT_DEMO_DETAIL.source_path)).toBeInTheDocument();
    expect(screen.getByText(String(HOLDOUT_DEMO_DETAIL.n_dev_bars))).toBeInTheDocument();
    expect(screen.getByText(String(HOLDOUT_DEMO_DETAIL.n_holdout_bars))).toBeInTheDocument();

    const sourceLink = screen.getByRole("link", { name: RUN_SPY.experiment_id });
    expect(sourceLink).toHaveAttribute("href", `/runs/${RUN_SPY.experiment_id}`);
  });

  it("renders the source ID as plain text when source_kind is not 'run'", () => {
    const hpoSourced = {
      ...HOLDOUT_DEMO_DETAIL,
      source_kind: "hpo" as const,
      source_id: "trial_42",
    };
    render(
      <MemoryRouter future={ROUTER_FUTURE_FLAGS}>
        <DevVsHoldoutPanel holdout={hpoSourced} />
      </MemoryRouter>,
    );

    expect(screen.queryByRole("link", { name: "trial_42" })).not.toBeInTheDocument();
    expect(screen.getByText("trial_42")).toBeInTheDocument();
  });
});
