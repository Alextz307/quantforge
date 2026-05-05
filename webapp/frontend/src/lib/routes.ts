export const ROUTES = {
  login: "/login",
  runs: "/runs",
  runDetail: "/runs/:experimentId",
  comparisons: "/comparisons",
  comparisonDetail: "/comparisons/:name",
  holdout: "/holdout",
  holdoutDetail: "/holdout/:name",
  regime: "/regime",
  regimeDetail: "/regime/:name",
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

export function regimeDetailPath(name: string): string {
  return `/regime/${encodeURIComponent(name)}`;
}

export function studyDetailPath(name: string): string {
  return `/studies/${encodeURIComponent(name)}`;
}

export function hpoDetailPath(name: string): string {
  return `/hpo/${encodeURIComponent(name)}`;
}

export const FROM_QUERY_PARAM = "from";

export function resolveFromParam(params: URLSearchParams): string {
  const from = params.get(FROM_QUERY_PARAM);
  return from && from.startsWith("/") ? from : ROUTES.runs;
}
