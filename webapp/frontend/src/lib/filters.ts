export const ALL_OPTION = "__all__";

export function uniqSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values)).sort();
}

export function withActiveOption<T extends string>(
  present: readonly T[],
  active: T | typeof ALL_OPTION,
): T[] {
  // Keep the active filter value selectable even when no row on the current
  // page carries it (a shared ?filter= URL, or a value paged out of view), so
  // the dropdown doesn't silently snap back to "All".
  if (active !== ALL_OPTION && !present.includes(active)) {
    return uniqSorted([...present, active]) as T[];
  }
  return [...present];
}

export function readValidSince(raw: string | null): string {
  // Drop an unparseable ?since so it is never forwarded to the server (which
  // would 422 on a bad datetime); an empty string disables the filter.
  if (raw === null || raw === "") return "";
  return Number.isNaN(new Date(raw).getTime()) ? "" : raw;
}

// Read + validate ``?sort_by``/``?order`` from the URL, falling back to
// ``fallback`` for an unknown column or order. Shared by every sortable list
// page so the validation lives once.
export function readSortState<K extends string>(
  params: URLSearchParams,
  keys: ReadonlySet<K>,
  fallback: { sortBy: K; order: "asc" | "desc" },
): { sortBy: K; order: "asc" | "desc" } {
  const sortBy = params.get("sort_by");
  const order = params.get("order");
  return {
    sortBy: sortBy !== null && keys.has(sortBy as K) ? (sortBy as K) : fallback.sortBy,
    order: order === "asc" || order === "desc" ? order : fallback.order,
  };
}

// The ``setParams`` update for clicking a sortable header: the active column
// flips desc<->asc (starting desc), a different column starts desc.
export function toggleSortParams<K extends string>(
  current: { sortBy: K; order: "asc" | "desc" },
  col: K,
): Record<string, string> {
  const order = current.sortBy === col && current.order === "desc" ? "asc" : "desc";
  return { sort_by: col, order };
}
