export const ROUTES = {
  login: "/login",
  configure: "/configure",
  configureRun: "/configure/run",
  configureTune: "/configure/tune",
  configureCompare: "/configure/compare",
  configureHoldout: "/configure/holdout",
  jobs: "/jobs",
  jobDetail: "/jobs/:jobId",
  runs: "/runs",
  runDetail: "/runs/:experimentId",
  comparisons: "/comparisons",
  comparisonDetail: "/comparisons/:name",
  holdout: "/holdout",
  holdoutDetail: "/holdout/:name",
  studies: "/studies",
  studyDetail: "/studies/:name",
  hpo: "/hpo",
  hpoDetail: "/hpo/:name",
  admin: "/admin",
} as const;

export function runDetailPath(experimentId: string): string {
  return `/runs/${encodeURIComponent(experimentId)}`;
}

export function comparisonDetailPath(name: string): string {
  return `/comparisons/${encodeURIComponent(name)}`;
}

export function holdoutDetailPath(name: string): string {
  return `/holdout/${encodeURIComponent(name)}`;
}

export function studyDetailPath(name: string): string {
  return `/studies/${encodeURIComponent(name)}`;
}

export function hpoDetailPath(name: string): string {
  return `/hpo/${encodeURIComponent(name)}`;
}

export function jobDetailPath(jobId: string): string {
  return `/jobs/${encodeURIComponent(jobId)}`;
}

export const FROM_QUERY_PARAM = "from";

export function resolveFromParam(params: URLSearchParams): string {
  const from = params.get(FROM_QUERY_PARAM);
  return from && from.startsWith("/") ? from : ROUTES.runs;
}
