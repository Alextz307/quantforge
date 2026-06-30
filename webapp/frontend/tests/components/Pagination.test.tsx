import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Pagination } from "@/components/Pagination";
import { renderWithProviders } from "../util/render";

const TOTAL = 120;
const LIMIT = 50;
const FULL_PAGE_COUNT = 50;
const PAGE_TWO_OFFSET = 50;
const LAST_PAGE_OFFSET = 100;
const LAST_PAGE_COUNT = 20;
const SHRUNK_TOTAL = 10;

describe("Pagination", () => {
  it("shows the current range and total", () => {
    renderWithProviders(
      <Pagination
        total={TOTAL}
        limit={LIMIT}
        offset={0}
        count={FULL_PAGE_COUNT}
        onOffset={vi.fn()}
      />,
    );
    expect(screen.getByText(/Showing 1-50 of 120/)).toBeInTheDocument();
  });

  it("disables Previous on the first page and enables Next", () => {
    renderWithProviders(
      <Pagination
        total={TOTAL}
        limit={LIMIT}
        offset={0}
        count={FULL_PAGE_COUNT}
        onOffset={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /previous/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /next/i })).toBeEnabled();
  });

  it("disables Next on the last (partial) page", () => {
    renderWithProviders(
      <Pagination
        total={TOTAL}
        limit={LIMIT}
        offset={LAST_PAGE_OFFSET}
        count={LAST_PAGE_COUNT}
        onOffset={vi.fn()}
      />,
    );
    expect(screen.getByText(/Showing 101-120 of 120/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /previous/i })).toBeEnabled();
  });

  it("advances by one page on Next and back on Previous", async () => {
    const user = userEvent.setup();
    const onOffset = vi.fn();
    renderWithProviders(
      <Pagination
        total={TOTAL}
        limit={LIMIT}
        offset={PAGE_TWO_OFFSET}
        count={FULL_PAGE_COUNT}
        onOffset={onOffset}
      />,
    );
    await user.click(screen.getByRole("button", { name: /next/i }));
    expect(onOffset).toHaveBeenCalledWith(PAGE_TWO_OFFSET + LIMIT);
    await user.click(screen.getByRole("button", { name: /previous/i }));
    expect(onOffset).toHaveBeenCalledWith(PAGE_TWO_OFFSET - LIMIT);
  });

  it("keeps Previous reachable on an empty out-of-range page", () => {
    // Dead-end guard: an offset past a shrunken total yields count 0, but
    // Previous must stay enabled so the user can navigate back into range.
    renderWithProviders(
      <Pagination
        total={SHRUNK_TOTAL}
        limit={LIMIT}
        offset={PAGE_TWO_OFFSET}
        count={0}
        onOffset={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /previous/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
  });
});
