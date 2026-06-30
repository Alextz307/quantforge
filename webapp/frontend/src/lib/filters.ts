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
