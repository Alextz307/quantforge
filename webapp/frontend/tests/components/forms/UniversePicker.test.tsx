import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { UniversePicker } from "@/components/forms/UniversePicker";
import { renderWithProviders } from "../../util/render";

describe("UniversePicker", () => {
  it("loads the configured universes and applies the selected preset", async () => {
    const onApply = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<UniversePicker onApply={onApply} />);

    // Wait for the configs/universe list query to populate the <option>s
    // before user-event tries to select. Without this, the select is still
    // in its loading-disabled state.
    await screen.findByRole("option", { name: "spy_daily_5y" });
    const select = screen.getByLabelText(/universe preset/i);
    await user.selectOptions(select, "spy_daily_5y");
    const apply = await screen.findByRole("button", { name: /Apply preset/i });
    await vi.waitFor(() => {
      expect(apply).not.toBeDisabled();
    });
    await user.click(apply);

    expect(onApply).toHaveBeenCalledWith({
      tickers: "SPY",
      start: "2020-01-01",
      end: "2024-12-31",
      interval: "daily",
    });
  });
});
