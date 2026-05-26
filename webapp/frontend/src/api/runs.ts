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
import { queryKeys, type RunsPageParams } from "./queryKeys";

export type RunSummary = components["schemas"]["RunSummary"];
export type RunDetail = components["schemas"]["RunDetail"];
export type FoldRow = components["schemas"]["FoldRow"];
export type RunsPage = components["schemas"]["RunsPage"];
export type RunSortBy = components["schemas"]["RunSortBy"];
export type SortOrder = components["schemas"]["SortOrder"];

const LIST_STALE_TIME = 30_000;
// Free-form filter inputs spawn one query key per debounced state. Cap
// gcTime so stale entries (e.g. abandoned mid-search keys) don't pin the
// React Query cache; the default 5min is too generous for unbounded keys.
const LIST_GC_TIME = 60_000;

function runsPageConfig(params: RunsPageParams): ApiQueryOptions<RunsPage> {
  return {
    queryKey: queryKeys.runsPage(params),
    fetcher: () =>
      apiClient.GET(API_PATHS.runs, {
        params: {
          query: {
            limit: params.limit,
            offset: params.offset,
            sort_by: params.sortBy,
            order: params.order,
            ...(params.strategy !== undefined ? { strategy: params.strategy } : {}),
            ...(params.ticker !== undefined ? { ticker: params.ticker } : {}),
            ...(params.since !== undefined ? { since: params.since } : {}),
          },
        },
      }),
    errorMsg: "Failed to load runs",
    staleTime: LIST_STALE_TIME,
    gcTime: LIST_GC_TIME,
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

export function useRunsPage(params: RunsPageParams): UseQueryResult<RunsPage> {
  return useApiQuery(runsPageConfig(params));
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
