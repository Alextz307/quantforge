import createClient, { type Middleware } from "openapi-fetch";
import {
  useQuery,
  type DefaultError,
  type Query,
  type QueryClient,
  type QueryKey,
  type UseQueryResult,
} from "@tanstack/react-query";
import { FROM_QUERY_PARAM, ROUTES } from "@/lib/routes";
import { extractApiError } from "./errors";
import type { paths } from "./generated/schema";

const UNAUTHORIZED = 401;
const AUTH_PATH_PREFIX = "/api/auth/";

const redirectOn401: Middleware = {
  onResponse({ response, request }) {
    if (response.status !== UNAUTHORIZED) return;
    if (typeof window === "undefined") return;
    if (window.location.pathname === ROUTES.login) return;
    if (new URL(request.url).pathname.startsWith(AUTH_PATH_PREFIX)) return;
    const from = window.location.pathname + window.location.search;
    const target = `${ROUTES.login}?${FROM_QUERY_PARAM}=${encodeURIComponent(from)}`;
    window.location.assign(target);
  },
};

const baseUrl = typeof window === "undefined" ? "http://localhost" : window.location.origin;

export const apiClient = createClient<paths>({
  baseUrl,
  credentials: "include",
  fetch: (...args) => globalThis.fetch(...args),
});

apiClient.use(redirectOn401);

export type { paths } from "./generated/schema";
export type { components } from "./generated/schema";

interface ApiResponse<T> {
  data?: T;
  error?: unknown;
  response: { ok: boolean };
}

type Fetcher<T> = () => Promise<ApiResponse<T>>;

async function runFetch<T>(fetcher: Fetcher<T>, errorMsg: string): Promise<T> {
  const { data, error, response } = await fetcher();
  if (!response.ok || !data) throw new Error(extractApiError(error, errorMsg));
  return data;
}

type RefetchIntervalFn<T> = (query: Query<T, DefaultError, T>) => number | false | undefined;

export interface ApiQueryOptions<T> {
  queryKey: QueryKey;
  fetcher: Fetcher<T>;
  errorMsg: string;
  staleTime?: number;
  gcTime?: number;
  refetchInterval?: number | false | RefetchIntervalFn<T>;
  enabled?: boolean;
}

export function useApiQuery<T>(opts: ApiQueryOptions<T>): UseQueryResult<T> {
  return useQuery({
    queryKey: opts.queryKey,
    queryFn: () => runFetch(opts.fetcher, opts.errorMsg),
    ...(opts.staleTime !== undefined ? { staleTime: opts.staleTime } : {}),
    ...(opts.gcTime !== undefined ? { gcTime: opts.gcTime } : {}),
    ...(opts.refetchInterval !== undefined ? { refetchInterval: opts.refetchInterval } : {}),
    ...(opts.enabled !== undefined ? { enabled: opts.enabled } : {}),
  });
}

export function prefetchApiQuery<T>(
  queryClient: QueryClient,
  opts: ApiQueryOptions<T>,
): Promise<void> {
  return queryClient.prefetchQuery({
    queryKey: opts.queryKey,
    queryFn: () => runFetch(opts.fetcher, opts.errorMsg),
    ...(opts.staleTime !== undefined ? { staleTime: opts.staleTime } : {}),
  });
}
