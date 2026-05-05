export const ALL_OPTION = "__all__";

export function uniqSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values)).sort();
}
