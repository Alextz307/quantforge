import { useCallback } from "react";
import { useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  listPageConfig,
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
export type FeatureImportanceResponse = components["schemas"]["FeatureImportanceResponse"];
export type FeatureImportanceEntry = components["schemas"]["FeatureImportanceEntry"];
export type ImportanceMethod = components["schemas"]["ImportanceMethod"];

export interface RunsListOptions {
  allUsers?: boolean;
  // Gate the fetch (React Query ``enabled``). Used by the holdout-source picker
  // so the runs list isn't fetched while the HPO source tab is selected.
  enabled?: boolean;
}

function runsPageConfig(params: RunsPageParams, opts: RunsListOptions): ApiQueryOptions<RunsPage> {
  const allUsers = opts.allUsers ?? false;
  return listPageConfig({
    queryKey: queryKeys.runsPage({ ...params, allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.runs, {
        params: {
          query: {
            // openapi-fetch omits null/undefined query values; `?? null` keeps
            // exactOptionalPropertyTypes satisfied while dropping absent filters.
            limit: params.limit,
            offset: params.offset,
            sort_by: params.sortBy,
            order: params.order,
            strategy: params.strategy ?? null,
            ticker: params.ticker ?? null,
            since: params.since ?? null,
            ...(allUsers ? { all: true } : {}),
          },
        },
      }),
    errorMsg: "Failed to load runs",
    ...(opts.enabled !== undefined ? { enabled: opts.enabled } : {}),
  });
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

export function useRunsPage(
  params: RunsPageParams,
  opts: RunsListOptions = {},
): UseQueryResult<RunsPage> {
  return useApiQuery(runsPageConfig(params, opts));
}

export function useRun(experimentId: string): UseQueryResult<RunDetail> {
  return useApiQuery(runConfig(experimentId));
}

export function useRunFolds(experimentId: string): UseQueryResult<FoldRow[]> {
  return useApiQuery(runFoldsConfig(experimentId));
}

function runFeatureImportanceConfig(
  experimentId: string,
): ApiQueryOptions<FeatureImportanceResponse> {
  return {
    queryKey: queryKeys.runFeatureImportance(experimentId),
    fetcher: () =>
      apiClient.GET(API_PATHS.runFeatureImportance, {
        params: { path: { experiment_id: experimentId } },
      }),
    errorMsg: "Failed to load feature importance",
    staleTime: Infinity,
  };
}

export function useFeatureImportance(
  experimentId: string,
): UseQueryResult<FeatureImportanceResponse> {
  return useApiQuery(runFeatureImportanceConfig(experimentId));
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
