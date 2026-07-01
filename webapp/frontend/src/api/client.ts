import createClient, { type Middleware } from "openapi-fetch";
import {
  keepPreviousData,
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

// The largest page the list endpoints accept (their FastAPI ``Query(le=...)``
// cap). Pagination clamps a user-supplied ``?limit`` to this, and the full-list
// pickers request exactly this many rows.
export const MAX_PAGE_LIMIT = 500;

// Cache tuning shared by every list endpoint: a short stale window keeps
// paginated lists fresh, and capped GC retention bounds the many short-lived
// keys that paging/sort/filter combinations spawn.
export const LIST_STALE_TIME = 30_000;
export const LIST_GC_TIME = 60_000;

// A list endpoint's page projected to the rows a full-list picker renders plus
// the pre-slice total, so a picker can flag when its MAX_PAGE_LIMIT window
// truncates the set (total > items.length).
export interface PickerPage<T> {
  items: T[];
  total: number;
}

interface ApiResponse<T> {
  data?: T;
  error?: unknown;
  response: { ok: boolean };
}

export type Fetcher<T> = () => Promise<ApiResponse<T>>;

async function runFetch<T>(fetcher: Fetcher<T>, errorMsg: string): Promise<T> {
  const { data, error, response } = await fetcher();
  if (!response.ok || !data) throw new Error(extractApiError(error, errorMsg));
  return data;
}

type RefetchIntervalFn<T> = (query: Query<T, DefaultError, T>) => number | false | undefined;

export interface ApiQueryOptions<T, S = T> {
  queryKey: QueryKey;
  fetcher: Fetcher<T>;
  errorMsg: string;
  staleTime?: number;
  gcTime?: number;
  refetchInterval?: number | false | RefetchIntervalFn<T>;
  enabled?: boolean;
  // Post-fetch transform (React Query ``select``). Lets a paginated endpoint
  // back a full-list convenience hook by projecting the page down to its items.
  select?: (data: T) => S;
  // Keep the previous page's data rendered while the next page/sort/filter
  // request is in flight, instead of dropping to the loading fallback and
  // flashing an empty table on every interaction.
  keepPreviousPage?: boolean;
}

export function useApiQuery<T, S = T>(opts: ApiQueryOptions<T, S>): UseQueryResult<S> {
  return useQuery<T, DefaultError, S>({
    queryKey: opts.queryKey,
    queryFn: () => runFetch(opts.fetcher, opts.errorMsg),
    ...(opts.select !== undefined ? { select: opts.select } : {}),
    ...(opts.staleTime !== undefined ? { staleTime: opts.staleTime } : {}),
    ...(opts.gcTime !== undefined ? { gcTime: opts.gcTime } : {}),
    ...(opts.refetchInterval !== undefined ? { refetchInterval: opts.refetchInterval } : {}),
    ...(opts.enabled !== undefined ? { enabled: opts.enabled } : {}),
    ...(opts.keepPreviousPage ? { placeholderData: keepPreviousData } : {}),
  });
}

// Shared shape for every paginated list-page query: the short-stale/capped-GC
// cache tuning plus keepPreviousPage, so each ``*PageConfig`` builder supplies
// only what differs (query key, typed fetcher, error message, optional enable
// gate). The picker configs deliberately opt out - they use ``select`` + an
// infinite-ish stale window, not this.
export function listPageConfig<TPage>(opts: {
  queryKey: QueryKey;
  fetcher: Fetcher<TPage>;
  errorMsg: string;
  enabled?: boolean;
}): ApiQueryOptions<TPage> {
  return {
    queryKey: opts.queryKey,
    fetcher: opts.fetcher,
    errorMsg: opts.errorMsg,
    staleTime: LIST_STALE_TIME,
    gcTime: LIST_GC_TIME,
    keepPreviousPage: true,
    ...(opts.enabled !== undefined ? { enabled: opts.enabled } : {}),
  };
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
