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
  regimeReports: "/api/regime-reports",
  regimeReport: "/api/regime-reports/{name}",
  regimePlot: "/api/regime-reports/{name}/plots/{plot_name}",
  studies: "/api/studies",
  study: "/api/studies/{name}",
  studyConsolidated: "/api/studies/{name}/consolidated",
  studyConsolidatedPlot: "/api/studies/{name}/consolidated/plots/{plot_name}",
  studyConsolidatedTable: "/api/studies/{name}/consolidated/tables/{table_name}",
  hpoStudies: "/api/hpo",
  hpoStudy: "/api/hpo/{name}",
  hpoTrials: "/api/hpo/{name}/trials",
  jobs: "/api/jobs",
  job: "/api/jobs/{job_id}",
  jobLog: "/api/jobs/{job_id}/log",
  jobStream: "/api/jobs/{job_id}/stream",
  configs: "/api/configs/{kind}",
  configDetail: "/api/configs/{kind}/{name}",
  configValidate: "/api/configs/validate",
  strategies: "/api/strategies",
  strategySchema: "/api/strategies/{name}/schema",
  publicSettings: "/api/settings/public",
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
