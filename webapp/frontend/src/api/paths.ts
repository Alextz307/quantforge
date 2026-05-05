export const API_PATHS = {
  runs: "/api/runs",
  run: "/api/runs/{experiment_id}",
  runFolds: "/api/runs/{experiment_id}/folds",
  runPlot: "/api/runs/{experiment_id}/plots/{plot_name}",
  comparisons: "/api/comparisons",
  comparison: "/api/comparisons/{name}",
  comparisonPlot: "/api/comparisons/{name}/plots/{plot_name}",
  holdoutEvals: "/api/holdout-evals",
  holdoutEval: "/api/holdout-evals/{name}",
  holdoutEvalPlot: "/api/holdout-evals/{name}/plots/{plot_name}",
} as const;

// MSW v2 expects `:name` path params; openapi-fetch (matching FastAPI's path
// template) uses `{name}`. Same param name on both sides keeps tests aligned
// with production routes — change the API path here and MSW handlers track it.
export function toMswPath(path: string): string {
  return path.replace(/\{(\w+)\}/g, ":$1");
}

export function fillPath(template: string, params: Readonly<Record<string, string>>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => {
    const value = params[key];
    if (value === undefined) throw new Error(`Missing path param '${key}' for '${template}'`);
    return encodeURIComponent(value);
  });
}
