import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { renderWithProviders } from "../util/render";

interface Row {
  id: string;
  name: string;
  category: string;
  count: number;
}

interface Filters {
  category: string;
}

const ROWS: readonly Row[] = [
  { id: "a", name: "Alpha", category: "x", count: 1 },
  { id: "b", name: "Bravo", category: "y", count: 2 },
  { id: "c", name: "Charlie", category: "x", count: 3 },
];

const ALL = "__all__";

function applyFilters(rows: readonly Row[], f: Filters): readonly Row[] {
  return rows.filter((r) => f.category === ALL || r.category === f.category);
}

function Harness() {
  const [category, setCategory] = useState<string>(ALL);
  return (
    <FilterableTablePage<Row, Filters>
      rows={ROWS}
      filters={{ category }}
      applyFilters={applyFilters}
      filterControls={
        <label>
          Category
          <select
            value={category}
            onChange={(e) => {
              setCategory(e.target.value);
            }}
          >
            <option value={ALL}>All</option>
            <option value="x">x</option>
            <option value="y">y</option>
          </select>
        </label>
      }
      rowKey={(r) => r.id}
      rowName={(r) => r.name}
      rowHref={(r) => `/rows/${r.id}`}
      tableTestId="rows-table"
      emptyMessage="Nothing to show."
      columns={[
        { header: "Category", cellClassName: "font-mono", render: (r) => r.category },
        { header: "Count", align: "right", render: (r) => String(r.count) },
      ]}
    />
  );
}

function TreeWithDetail() {
  return (
    <Routes>
      <Route path="/" element={<Harness />} />
      <Route path="/rows/:id" element={<div>row detail</div>} />
    </Routes>
  );
}

describe("FilterableTablePage", () => {
  it("renders all rows under the implicit Name column with detail links", () => {
    renderWithProviders(<TreeWithDetail />);
    const table = screen.getByTestId("rows-table");
    expect(within(table).getByRole("columnheader", { name: "Name" })).toBeInTheDocument();
    expect(within(table).getByRole("link", { name: "Alpha" })).toHaveAttribute("href", "/rows/a");
    expect(within(table).getByRole("link", { name: "Bravo" })).toBeInTheDocument();
    expect(within(table).getByRole("link", { name: "Charlie" })).toBeInTheDocument();
  });

  it("filters via applyFilters when filter state changes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TreeWithDetail />);
    await user.selectOptions(screen.getByLabelText(/category/i), "y");

    const table = screen.getByTestId("rows-table");
    expect(within(table).queryByText("Alpha")).not.toBeInTheDocument();
    expect(within(table).getByText("Bravo")).toBeInTheDocument();
    expect(within(table).queryByText("Charlie")).not.toBeInTheDocument();
  });

  it("renders the empty message when applyFilters returns no rows", () => {
    function EmptyHarness() {
      return (
        <FilterableTablePage<Row, Filters>
          rows={ROWS}
          filters={{ category: "z" }}
          applyFilters={applyFilters}
          filterControls={null}
          rowKey={(r) => r.id}
          rowName={(r) => r.name}
          rowHref={(r) => `/rows/${r.id}`}
          tableTestId="rows-table"
          emptyMessage="Nothing to show."
          columns={[]}
        />
      );
    }
    renderWithProviders(
      <Routes>
        <Route path="/" element={<EmptyHarness />} />
      </Routes>,
    );
    expect(screen.getByText("Nothing to show.")).toBeInTheDocument();
    expect(screen.queryByTestId("rows-table")).not.toBeInTheDocument();
  });

  it("navigates to rowHref when the name link is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TreeWithDetail />);
    await user.click(screen.getByRole("link", { name: "Alpha" }));
    expect(await screen.findByText("row detail")).toBeInTheDocument();
  });

  it("calls rowOnHover with the hovered row", async () => {
    const calls: Row[] = [];
    function HoverHarness() {
      return (
        <FilterableTablePage<Row, Filters>
          rows={ROWS}
          filters={{ category: ALL }}
          applyFilters={applyFilters}
          filterControls={null}
          rowKey={(r) => r.id}
          rowName={(r) => r.name}
          rowHref={(r) => `/rows/${r.id}`}
          rowOnHover={(r) => calls.push(r)}
          tableTestId="rows-table"
          emptyMessage="empty"
          columns={[]}
        />
      );
    }
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path="/" element={<HoverHarness />} />
      </Routes>,
    );
    await user.hover(screen.getByRole("link", { name: "Bravo" }));
    expect(calls.map((r) => r.id)).toContain("b");
  });
});
