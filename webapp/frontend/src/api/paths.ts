export const API_PATHS = {
  runs: "/api/runs",
  run: "/api/runs/{experiment_id}",
  runFolds: "/api/runs/{experiment_id}/folds",
  runPlot: "/api/runs/{experiment_id}/plots/{plot_name}",
} as const;

// MSW v2 expects `:name` path params; openapi-fetch (matching FastAPI's path
// template) uses `{name}`. Same param name on both sides keeps tests aligned
// with production routes — change the API path here and MSW handlers track it.
export function toMswPath(path: string): string {
  return path.replace(/\{(\w+)\}/g, ":$1");
}
