import { useMemo, type ReactNode } from "react";
import { Link } from "react-router-dom";

export interface FilterableTableColumn<TRow> {
  header: string;
  align?: "left" | "right";
  cellClassName?: string;
  render: (row: TRow) => ReactNode;
}

export interface FilterableTablePageProps<TRow, TFilters> {
  rows: readonly TRow[];
  filters: TFilters;
  applyFilters: (rows: readonly TRow[], filters: TFilters) => readonly TRow[];
  filterControls: ReactNode;
  filterGridClassName?: string;
  rowKey: (row: TRow) => string;
  rowName: (row: TRow) => string;
  rowHref: (row: TRow) => string;
  rowOnHover?: (row: TRow) => void;
  columns: readonly FilterableTableColumn<TRow>[];
  emptyMessage: string;
  tableTestId: string;
  nameHeader?: string;
}

function cellClass(isLast: boolean, align: "left" | "right", extra?: string): string {
  const padRight = isLast ? "pr-0" : "pr-4";
  const alignCls = align === "right" ? " text-right" : "";
  const base = `py-2 ${padRight}${alignCls}`;
  return extra ? `${base} ${extra}` : base;
}

export function FilterableTablePage<TRow, TFilters>({
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
}: FilterableTablePageProps<TRow, TFilters>) {
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
                {columns.map((c, i) => (
                  <th
                    key={c.header}
                    className={cellClass(i === columns.length - 1, c.align ?? "left")}
                  >
                    {c.header}
                  </th>
                ))}
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
