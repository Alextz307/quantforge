import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RunsPage } from "@/pages/RunsPage";
import { ROUTES } from "@/lib/routes";
import { RUN_IVV_VOO, RUN_SPY } from "../msw/handlers";
import { renderWithProviders } from "../util/render";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.runs} element={<RunsPage />} />
      <Route path={ROUTES.runDetail} element={<div>detail page</div>} />
    </Routes>
  );
}

describe("RunsPage", () => {
  it("lists every run returned by the API", async () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.runs] });

    expect(await screen.findByRole("link", { name: RUN_SPY.name })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: RUN_IVV_VOO.name })).toBeInTheDocument();
  });

  it("filters the table down to the selected strategy", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.runs] });

    await screen.findByRole("link", { name: RUN_SPY.name });
    await user.selectOptions(screen.getByLabelText(/strategy/i), RUN_IVV_VOO.strategy);

    const table = screen.getByTestId("runs-table");
    expect(within(table).queryByText(RUN_SPY.name)).not.toBeInTheDocument();
    expect(within(table).getByText(RUN_IVV_VOO.name)).toBeInTheDocument();
  });

  it("filters by ticker", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.runs] });

    await screen.findByRole("link", { name: RUN_SPY.name });
    await user.selectOptions(screen.getByLabelText(/ticker/i), "VOO");

    const table = screen.getByTestId("runs-table");
    expect(within(table).queryByText(RUN_SPY.name)).not.toBeInTheDocument();
    expect(within(table).getByText(RUN_IVV_VOO.name)).toBeInTheDocument();
  });

  it("navigates to the detail page when a row link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.runs] });

    const link = await screen.findByRole("link", { name: RUN_SPY.name });
    await user.click(link);

    expect(await screen.findByText("detail page")).toBeInTheDocument();
  });
});
