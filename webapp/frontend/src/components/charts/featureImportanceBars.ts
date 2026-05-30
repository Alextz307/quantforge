import type { FeatureImportanceResponse, ImportanceMethod } from "@/api/runs";

export interface FeatureImportanceBars {
  features: string[];
  values: number[];
  errors: number[] | null;
}

export function buildFeatureImportanceBars(
  entries: FeatureImportanceResponse["entries"],
  method: ImportanceMethod,
): FeatureImportanceBars {
  // Ascending so Plotly's bottom-up categorical draw puts the largest bar on top;
  // keep features/values/errors index-aligned.
  const rows = entries
    .filter((e) => e.method === method && e.importance !== null)
    .map((e) => ({ feature: e.feature, importance: e.importance as number, std: e.std }))
    .sort((a, b) => a.importance - b.importance);
  // A single-fold run reports std=0.0 (a placeholder, not a measurement), so
  // gate on std > 0 to avoid drawing meaningless zero-height whiskers.
  const hasError = method === "permutation" && rows.some((r) => r.std !== null && r.std > 0);
  return {
    features: rows.map((r) => r.feature),
    values: rows.map((r) => r.importance),
    errors: hasError ? rows.map((r) => r.std ?? 0) : null,
  };
}
