import { useCallback } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  prefetchApiQuery,
  useApiQuery,
  type ApiQueryOptions,
  type components,
} from "./client";
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";

export type HpoSummary = components["schemas"]["HpoSummary"];
export type HpoDetail = components["schemas"]["HpoDetail"];
export type TrialRow = components["schemas"]["TrialRow"];
export type StudyDirection = components["schemas"]["StudyDirection"];

const LIST_STALE_TIME = 30_000;

function hpoStudiesConfig(): ApiQueryOptions<HpoSummary[]> {
  return {
    queryKey: queryKeys.hpoStudies,
    fetcher: () => apiClient.GET(API_PATHS.hpoStudies),
    errorMsg: "Failed to load HPO studies",
    staleTime: LIST_STALE_TIME,
  };
}

function hpoStudyConfig(name: string): ApiQueryOptions<HpoDetail> {
  return {
    queryKey: queryKeys.hpoStudy(name),
    fetcher: () => apiClient.GET(API_PATHS.hpoStudy, { params: { path: { name } } }),
    errorMsg: "Failed to load HPO study",
    staleTime: Infinity,
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

export function useHpoStudies(): UseQueryResult<HpoSummary[]> {
  return useApiQuery(hpoStudiesConfig());
}

export function useHpoStudy(name: string): UseQueryResult<HpoDetail> {
  return useApiQuery(hpoStudyConfig(name));
}

export function useHpoTrials(name: string): UseQueryResult<TrialRow[]> {
  return useApiQuery(hpoTrialsConfig(name));
}

export function usePrefetchHpoStudy(): (name: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (name: string) => {
      void prefetchApiQuery(qc, hpoStudyConfig(name));
    },
    [qc],
  );
}
