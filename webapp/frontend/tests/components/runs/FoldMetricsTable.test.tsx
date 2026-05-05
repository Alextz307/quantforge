import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FoldMetricsTable } from "@/components/runs/FoldMetricsTable";
import { RUN_SPY_FOLDS } from "../../msw/handlers";

describe("FoldMetricsTable", () => {
  it("renders an empty message when no folds are supplied", () => {
    render(<FoldMetricsTable folds={[]} />);
    expect(screen.getByText(/no fold metrics available/i)).toBeInTheDocument();
  });

  it("renders one row per fold with formatted metrics", () => {
    render(<FoldMetricsTable folds={RUN_SPY_FOLDS} />);
    const table = screen.getByTestId("fold-metrics-table");
    const bodyRows = within(table).getAllByRole("row").slice(1);
    expect(bodyRows).toHaveLength(RUN_SPY_FOLDS.length);
    const [firstRow] = bodyRows;
    if (!firstRow) throw new Error("expected at least one fold row");
    expect(within(firstRow).getByText("1.1000")).toBeInTheDocument();
    expect(within(firstRow).getByText("15.00%")).toBeInTheDocument();
    expect(within(firstRow).getByText("42")).toBeInTheDocument();
  });
});
