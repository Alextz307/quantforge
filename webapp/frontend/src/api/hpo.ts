import { useCallback } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  listPageConfig,
  LIST_STALE_TIME,
  MAX_PAGE_LIMIT,
  prefetchApiQuery,
  useApiQuery,
  type ApiQueryOptions,
  type components,
  type PickerPage,
} from "./client";
import { API_PATHS, wsUrlFor } from "./paths";
import { queryKeys, type HpoStudiesPageParams } from "./queryKeys";

export type HpoSummary = components["schemas"]["HpoSummary"];
export type HpoDetail = components["schemas"]["HpoDetail"];
export type HpoStudiesPage = components["schemas"]["HpoStudiesPage"];
export type HpoSortBy = components["schemas"]["HpoSortBy"];
export type TrialRow = components["schemas"]["TrialRow"];
export type StudyDirection = components["schemas"]["StudyDirection"];
export type ParamImportanceResponse = components["schemas"]["ParamImportanceResponse"];

const STUDY_LIVE_REFETCH_MS = 3_000;
const IMPORTANCE_LIVE_REFETCH_MS = 30_000;

export interface HpoStudiesListOptions {
  allUsers?: boolean;
  // Gate the fetch (React Query ``enabled``). Used by the holdout-source picker
  // so the HPO list isn't fetched while the run source tab is selected.
  enabled?: boolean;
}

function hpoStudiesConfig(
  opts: HpoStudiesListOptions,
): ApiQueryOptions<HpoStudiesPage, PickerPage<HpoSummary>> {
  // The holdout-source picker wants the whole visible set; request the max page
  // and keep ``total`` so it can flag a set larger than that window.
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.hpoStudiesPicker({ allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.hpoStudies, {
        params: { query: { limit: MAX_PAGE_LIMIT, ...(allUsers ? { all: true } : {}) } },
      }),
    errorMsg: "Failed to load HPO studies",
    staleTime: LIST_STALE_TIME,
    select: (page) => ({ items: page.items, total: page.total }),
    ...(opts.enabled !== undefined ? { enabled: opts.enabled } : {}),
  };
}

function hpoStudiesPageConfig(
  params: HpoStudiesPageParams,
  opts: HpoStudiesListOptions,
): ApiQueryOptions<HpoStudiesPage> {
  const allUsers = opts.allUsers ?? false;
  return listPageConfig({
    queryKey: queryKeys.hpoStudiesPage({ ...params, allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.hpoStudies, {
        params: {
          query: {
            // openapi-fetch omits null/undefined query values; `?? null` keeps
            // exactOptionalPropertyTypes satisfied while dropping absent filters.
            limit: params.limit,
            offset: params.offset,
            sort_by: params.sortBy,
            order: params.order,
            store: params.store ?? null,
            since: params.since ?? null,
            ...(allUsers ? { all: true } : {}),
          },
        },
      }),
    errorMsg: "Failed to load HPO studies",
  });
}

function hpoStudyConfig(wireId: string, livePoll: boolean): ApiQueryOptions<HpoDetail> {
  return {
    queryKey: queryKeys.hpoStudy(wireId),
    fetcher: () => apiClient.GET(API_PATHS.hpoStudy, { params: { path: { wire_id: wireId } } }),
    errorMsg: "Failed to load HPO study",
    staleTime: Infinity,
    refetchInterval: livePoll
      ? (q) => (q.state.data?.live_job_id != null ? STUDY_LIVE_REFETCH_MS : false)
      : false,
  };
}

function hpoTrialsConfig(wireId: string): ApiQueryOptions<TrialRow[]> {
  return {
    queryKey: queryKeys.hpoTrials(wireId),
    fetcher: () => apiClient.GET(API_PATHS.hpoTrials, { params: { path: { wire_id: wireId } } }),
    errorMsg: "Failed to load HPO trials",
    staleTime: Infinity,
  };
}

function hpoParamImportanceConfig(
  wireId: string,
  isLive: boolean,
): ApiQueryOptions<ParamImportanceResponse> {
  return {
    queryKey: queryKeys.hpoParamImportance(wireId),
    fetcher: () =>
      apiClient.GET(API_PATHS.hpoParamImportance, { params: { path: { wire_id: wireId } } }),
    errorMsg: "Failed to load param importance",
    staleTime: Infinity,
    refetchInterval: isLive ? IMPORTANCE_LIVE_REFETCH_MS : false,
  };
}

export function useHpoStudiesPage(
  params: HpoStudiesPageParams,
  opts: HpoStudiesListOptions = {},
): UseQueryResult<HpoStudiesPage> {
  return useApiQuery(hpoStudiesPageConfig(params, opts));
}

export function useHpoStudies(
  opts: HpoStudiesListOptions = {},
): UseQueryResult<PickerPage<HpoSummary>> {
  return useApiQuery(hpoStudiesConfig(opts));
}

export function useHpoStudy(
  wireId: string,
  opts: { livePoll?: boolean } = {},
): UseQueryResult<HpoDetail> {
  return useApiQuery(hpoStudyConfig(wireId, opts.livePoll ?? false));
}

export function useHpoTrials(wireId: string): UseQueryResult<TrialRow[]> {
  return useApiQuery(hpoTrialsConfig(wireId));
}

export function useHpoParamImportance(
  wireId: string,
  opts: { isLive?: boolean } = {},
): UseQueryResult<ParamImportanceResponse> {
  return useApiQuery(hpoParamImportanceConfig(wireId, opts.isLive ?? false));
}

export function usePrefetchHpoStudy(): (wireId: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (wireId: string) => {
      void prefetchApiQuery(qc, hpoStudyConfig(wireId, false));
    },
    [qc],
  );
}

export function hpoStreamUrl(wireId: string, afterTrial?: number): string {
  return wsUrlFor(
    API_PATHS.hpoStream,
    { wire_id: wireId },
    afterTrial !== undefined ? { after_trial: afterTrial } : undefined,
  );
}
