export interface HpoImportanceRow {
  name: string;
  value: number;
}

export function buildHpoImportanceRows(importance: Record<string, number>): HpoImportanceRow[] {
  // Ascending so Plotly's bottom-up categorical draw puts the largest bar on top;
  // keep name/value index-aligned.
  return Object.entries(importance)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => a.value - b.value);
}
