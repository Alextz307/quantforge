import { useMemo, type ReactNode } from "react";
import { Link } from "react-router-dom";

export type SortOrder = "asc" | "desc";

export interface SortState<K extends string> {
  sortBy: K;
  order: SortOrder;
}

export interface FilterableTableColumn<TRow, K extends string = string> {
  header: string;
  align?: "left" | "right";
  cellClassName?: string;
  render: (row: TRow) => ReactNode;
  // When set, the column's header becomes a click-to-sort toggle. Existing
  // consumers omit this field and get a static header.
  sortKey?: K;
}

export interface FilterableTablePageProps<TRow, TFilters, K extends string = string> {
  rows: readonly TRow[];
  filters: TFilters;
  applyFilters: (rows: readonly TRow[], filters: TFilters) => readonly TRow[];
  filterControls: ReactNode;
  filterGridClassName?: string;
  rowKey: (row: TRow) => string;
  rowName: (row: TRow) => string;
  rowHref: (row: TRow) => string;
  rowOnHover?: (row: TRow) => void;
  columns: readonly FilterableTableColumn<TRow, K>[];
  emptyMessage: string;
  tableTestId: string;
  nameHeader?: string;
  // Sortable-header opt-in: both must be supplied for any column.sortKey to
  // render as a toggle. Sorting itself happens upstream (in ``applyFilters``);
  // these props only control the header indicator + click handler.
  sortState?: SortState<K>;
  onSortToggle?: (col: K) => void;
}

function cellClass(isLast: boolean, align: "left" | "right", extra?: string): string {
  const padRight = isLast ? "pr-0" : "pr-4";
  const alignCls = align === "right" ? " text-right" : "";
  const base = `py-2 ${padRight}${alignCls}`;
  return extra ? `${base} ${extra}` : base;
}

interface SortableHeaderProps<K extends string> {
  label: string;
  sortKey: K;
  state: SortState<K>;
  onToggle: (col: K) => void;
}

function SortableHeader<K extends string>({
  label,
  sortKey,
  state,
  onToggle,
}: SortableHeaderProps<K>): ReactNode {
  const active = state.sortBy === sortKey;
  const indicator = active ? (state.order === "desc" ? " ↓" : " ↑") : "";
  return (
    <button
      type="button"
      className={`hover:text-foreground ${active ? "text-foreground" : ""}`}
      onClick={() => {
        onToggle(sortKey);
      }}
    >
      {label}
      {indicator}
    </button>
  );
}

export function FilterableTablePage<TRow, TFilters, K extends string = string>({
  rows,
  filters,
  applyFilters,
  filterControls,
  filterGridClassName = "md:grid-cols-2",
  rowKey,
  rowName,
  rowHref,
  rowOnHover,
  columns,
  emptyMessage,
  tableTestId,
  nameHeader = "Name",
  sortState,
  onSortToggle,
}: FilterableTablePageProps<TRow, TFilters, K>) {
  const filtered = useMemo(() => applyFilters(rows, filters), [rows, filters, applyFilters]);
  const nameIsLast = columns.length === 0;

  return (
    <>
      <div className={`grid grid-cols-1 ${filterGridClassName} gap-4`}>{filterControls}</div>

      {filtered.length === 0 ? (
        <p className="text-sm text-muted-foreground">{emptyMessage}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid={tableTestId}>
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className={cellClass(nameIsLast, "left")}>{nameHeader}</th>
                {columns.map((c, i) => {
                  const isSortable =
                    c.sortKey !== undefined &&
                    sortState !== undefined &&
                    onSortToggle !== undefined;
                  return (
                    <th
                      key={c.header}
                      className={cellClass(i === columns.length - 1, c.align ?? "left")}
                    >
                      {isSortable ? (
                        <SortableHeader
                          label={c.header}
                          sortKey={c.sortKey as K}
                          state={sortState}
                          onToggle={onSortToggle}
                        />
                      ) : (
                        c.header
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => (
                <tr
                  key={rowKey(row)}
                  className="border-b last:border-0"
                  onMouseEnter={
                    rowOnHover
                      ? () => {
                          rowOnHover(row);
                        }
                      : undefined
                  }
                >
                  <td className={cellClass(nameIsLast, "left")}>
                    <Link to={rowHref(row)} className="text-primary hover:underline">
                      {rowName(row)}
                    </Link>
                  </td>
                  {columns.map((c, i) => (
                    <td
                      key={c.header}
                      className={cellClass(
                        i === columns.length - 1,
                        c.align ?? "left",
                        c.cellClassName,
                      )}
                    >
                      {c.render(row)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
