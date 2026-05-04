export const ROUTES = {
  login: "/login",
  runs: "/runs",
  admin: "/admin",
} as const;

export const FROM_QUERY_PARAM = "from";

export function resolveFromParam(params: URLSearchParams): string {
  const from = params.get(FROM_QUERY_PARAM);
  return from && from.startsWith("/") ? from : ROUTES.runs;
}
