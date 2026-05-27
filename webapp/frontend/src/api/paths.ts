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
  studies: "/api/studies",
  study: "/api/studies/{name}",
  studyConsolidated: "/api/studies/{name}/consolidated",
  studyConsolidatedPlot: "/api/studies/{name}/consolidated/plots/{plot_name}",
  studyConsolidatedTable: "/api/studies/{name}/consolidated/tables/{table_name}",
  studyStream: "/api/studies/{name}/stream",
  hpoStudies: "/api/hpo",
  hpoStudy: "/api/hpo/{wire_id}",
  hpoTrials: "/api/hpo/{wire_id}/trials",
  hpoParamImportance: "/api/hpo/{wire_id}/param-importance",
  hpoStream: "/api/hpo/{wire_id}/stream",
  jobs: "/api/jobs",
  job: "/api/jobs/{job_id}",
  jobLog: "/api/jobs/{job_id}/log",
  jobStream: "/api/jobs/{job_id}/stream",
  configs: "/api/configs/{kind}",
  configDetail: "/api/configs/{kind}/{name}",
  configValidate: "/api/configs/validate",
  studySpecSchema: "/api/configs/study_spec/schema",
  studySpecValidate: "/api/configs/study_spec/validate",
  studyUploads: "/api/configs/study/uploads",
  studyUpload: "/api/configs/study/uploads/{slug}",
  universeUploads: "/api/configs/universe/uploads",
  universeUpload: "/api/configs/universe/uploads/{slug}",
  universeSpecSchema: "/api/configs/universe_spec/schema",
  universeSpecValidate: "/api/configs/universe_spec/validate",
  strategies: "/api/strategies",
  strategySchema: "/api/strategies/{name}/schema",
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

export function wsUrlFor(
  template: string,
  params: Readonly<Record<string, string>>,
  query?: Readonly<Record<string, string | number>>,
): string {
  if (typeof window === "undefined") return "";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const path = fillPath(template, params);
  const qs = query
    ? "?" +
      Object.entries(query)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join("&")
    : "";
  return `${protocol}//${window.location.host}${path}${qs}`;
}
