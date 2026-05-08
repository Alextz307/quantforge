import { useCallback } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  prefetchApiQuery,
  useApiQuery,
  type ApiQueryOptions,
  type components,
} from "./client";
import { API_PATHS, wsUrlFor } from "./paths";
import { queryKeys } from "./queryKeys";

export type HpoSummary = components["schemas"]["HpoSummary"];
export type HpoDetail = components["schemas"]["HpoDetail"];
export type TrialRow = components["schemas"]["TrialRow"];
export type StudyDirection = components["schemas"]["StudyDirection"];
export type ParamImportanceResponse = components["schemas"]["ParamImportanceResponse"];

const LIST_STALE_TIME = 30_000;
const STUDY_LIVE_REFETCH_MS = 3_000;
const IMPORTANCE_LIVE_REFETCH_MS = 30_000;

function hpoStudiesConfig(): ApiQueryOptions<HpoSummary[]> {
  return {
    queryKey: queryKeys.hpoStudies,
    fetcher: () => apiClient.GET(API_PATHS.hpoStudies),
    errorMsg: "Failed to load HPO studies",
    staleTime: LIST_STALE_TIME,
  };
}

function hpoStudyConfig(name: string, livePoll: boolean): ApiQueryOptions<HpoDetail> {
  return {
    queryKey: queryKeys.hpoStudy(name),
    fetcher: () => apiClient.GET(API_PATHS.hpoStudy, { params: { path: { name } } }),
    errorMsg: "Failed to load HPO study",
    staleTime: Infinity,
    refetchInterval: livePoll
      ? (q) => (q.state.data?.live_job_id != null ? STUDY_LIVE_REFETCH_MS : false)
      : false,
  };
}

function hpoTrialsConfig(name: string): ApiQueryOptions<TrialRow[]> {
  return {
    queryKey: queryKeys.hpoTrials(name),
    fetcher: () => apiClient.GET(API_PATHS.hpoTrials, { params: { path: { name } } }),
    errorMsg: "Failed to load HPO trials",
    staleTime: Infinity,
  };
}

function hpoParamImportanceConfig(
  name: string,
  isLive: boolean,
): ApiQueryOptions<ParamImportanceResponse> {
  return {
    queryKey: queryKeys.hpoParamImportance(name),
    fetcher: () => apiClient.GET(API_PATHS.hpoParamImportance, { params: { path: { name } } }),
    errorMsg: "Failed to load param importance",
    staleTime: Infinity,
    refetchInterval: isLive ? IMPORTANCE_LIVE_REFETCH_MS : false,
  };
}

export function useHpoStudies(): UseQueryResult<HpoSummary[]> {
  return useApiQuery(hpoStudiesConfig());
}

export function useHpoStudy(
  name: string,
  opts: { livePoll?: boolean } = {},
): UseQueryResult<HpoDetail> {
  return useApiQuery(hpoStudyConfig(name, opts.livePoll ?? false));
}

export function useHpoTrials(name: string): UseQueryResult<TrialRow[]> {
  return useApiQuery(hpoTrialsConfig(name));
}

export function useHpoParamImportance(
  name: string,
  opts: { isLive?: boolean } = {},
): UseQueryResult<ParamImportanceResponse> {
  return useApiQuery(hpoParamImportanceConfig(name, opts.isLive ?? false));
}

export function usePrefetchHpoStudy(): (name: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (name: string) => {
      void prefetchApiQuery(qc, hpoStudyConfig(name, false));
    },
    [qc],
  );
}

export function hpoStreamUrl(name: string, afterTrial?: number): string {
  return wsUrlFor(
    API_PATHS.hpoStream,
    { name },
    afterTrial !== undefined ? { after_trial: afterTrial } : undefined,
  );
}
