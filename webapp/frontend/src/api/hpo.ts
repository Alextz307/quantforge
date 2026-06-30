import { useCallback } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  MAX_PAGE_LIMIT,
  prefetchApiQuery,
  useApiQuery,
  type ApiQueryOptions,
  type components,
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

const LIST_STALE_TIME = 30_000;
// Cap cache retention; paging/sort/filter combos spawn many short-lived keys.
const LIST_GC_TIME = 60_000;
const STUDY_LIVE_REFETCH_MS = 3_000;
const IMPORTANCE_LIVE_REFETCH_MS = 30_000;

export interface HpoStudiesListOptions {
  allUsers?: boolean;
}

function hpoStudiesConfig(
  opts: HpoStudiesListOptions,
): ApiQueryOptions<HpoStudiesPage, HpoSummary[]> {
  // The holdout-source picker wants the whole visible set; request the max page.
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.hpoStudiesPicker({ allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.hpoStudies, {
        params: { query: { limit: MAX_PAGE_LIMIT, ...(allUsers ? { all: true } : {}) } },
      }),
    errorMsg: "Failed to load HPO studies",
    staleTime: LIST_STALE_TIME,
    select: (page) => page.items,
  };
}

function hpoStudiesPageConfig(
  params: HpoStudiesPageParams,
  opts: HpoStudiesListOptions,
): ApiQueryOptions<HpoStudiesPage> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.hpoStudiesPage({ ...params, allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.hpoStudies, {
        params: {
          query: {
            limit: params.limit,
            offset: params.offset,
            sort_by: params.sortBy,
            order: params.order,
            ...(params.store !== undefined ? { store: params.store } : {}),
            ...(params.since !== undefined ? { since: params.since } : {}),
            ...(allUsers ? { all: true } : {}),
          },
        },
      }),
    errorMsg: "Failed to load HPO studies",
    staleTime: LIST_STALE_TIME,
    gcTime: LIST_GC_TIME,
  };
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

export function useHpoStudies(opts: HpoStudiesListOptions = {}): UseQueryResult<HpoSummary[]> {
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
