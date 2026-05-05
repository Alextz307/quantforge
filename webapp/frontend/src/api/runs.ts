import { useCallback } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  prefetchApiQuery,
  useApiQuery,
  type ApiQueryOptions,
  type components,
} from "./client";
import { API_PATHS, fillPath } from "./paths";
import { queryKeys } from "./queryKeys";

export type RunSummary = components["schemas"]["RunSummary"];
export type RunDetail = components["schemas"]["RunDetail"];
export type FoldRow = components["schemas"]["FoldRow"];

const LIST_STALE_TIME = 30_000;

function runsConfig(): ApiQueryOptions<RunSummary[]> {
  return {
    queryKey: queryKeys.runs,
    fetcher: () => apiClient.GET(API_PATHS.runs),
    errorMsg: "Failed to load runs",
    staleTime: LIST_STALE_TIME,
  };
}

function runConfig(experimentId: string): ApiQueryOptions<RunDetail> {
  return {
    queryKey: queryKeys.run(experimentId),
    fetcher: () =>
      apiClient.GET(API_PATHS.run, { params: { path: { experiment_id: experimentId } } }),
    errorMsg: "Failed to load run",
    staleTime: Infinity,
  };
}

function runFoldsConfig(experimentId: string): ApiQueryOptions<FoldRow[]> {
  return {
    queryKey: queryKeys.runFolds(experimentId),
    fetcher: () =>
      apiClient.GET(API_PATHS.runFolds, { params: { path: { experiment_id: experimentId } } }),
    errorMsg: "Failed to load folds",
    staleTime: Infinity,
  };
}

export function useRuns(): UseQueryResult<RunSummary[]> {
  return useApiQuery(runsConfig());
}

export function useRun(experimentId: string): UseQueryResult<RunDetail> {
  return useApiQuery(runConfig(experimentId));
}

export function useRunFolds(experimentId: string): UseQueryResult<FoldRow[]> {
  return useApiQuery(runFoldsConfig(experimentId));
}

export function usePrefetchRun(): (experimentId: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (experimentId: string) => {
      void prefetchApiQuery(qc, runConfig(experimentId));
    },
    [qc],
  );
}

export function plotDownloadUrl(experimentId: string, plotName: string): string {
  return fillPath(API_PATHS.runPlot, { experiment_id: experimentId, plot_name: plotName });
}
