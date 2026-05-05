export const ROUTES = {
  login: "/login",
  runs: "/runs",
  runDetail: "/runs/:experimentId",
  admin: "/admin",
} as const;

export function runDetailPath(experimentId: string): string {
  return `/runs/${encodeURIComponent(experimentId)}`;
}

export const FROM_QUERY_PARAM = "from";

export function resolveFromParam(params: URLSearchParams): string {
  const from = params.get(FROM_QUERY_PARAM);
  return from && from.startsWith("/") ? from : ROUTES.runs;
}
