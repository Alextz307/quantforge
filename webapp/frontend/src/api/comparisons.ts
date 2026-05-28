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

export type ComparisonSummary = components["schemas"]["ComparisonSummary"];
export type ComparisonDetail = components["schemas"]["ComparisonDetail"];
export type PerStrategyStatsRow = components["schemas"]["PerStrategyStatsRow"];

const LIST_STALE_TIME = 30_000;

export interface ComparisonsListOptions {
  allUsers?: boolean;
}

function comparisonsConfig(opts: ComparisonsListOptions): ApiQueryOptions<ComparisonSummary[]> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.comparisonsList(allUsers),
    fetcher: () =>
      allUsers
        ? apiClient.GET(API_PATHS.comparisons, { params: { query: { all: true } } })
        : apiClient.GET(API_PATHS.comparisons),
    errorMsg: "Failed to load comparisons",
    staleTime: LIST_STALE_TIME,
  };
}

function comparisonConfig(name: string): ApiQueryOptions<ComparisonDetail> {
  return {
    queryKey: queryKeys.comparison(name),
    fetcher: () => apiClient.GET(API_PATHS.comparison, { params: { path: { name } } }),
    errorMsg: "Failed to load comparison",
    staleTime: Infinity,
  };
}

export function useComparisons(
  opts: ComparisonsListOptions = {},
): UseQueryResult<ComparisonSummary[]> {
  return useApiQuery(comparisonsConfig(opts));
}

export function useComparison(name: string): UseQueryResult<ComparisonDetail> {
  return useApiQuery(comparisonConfig(name));
}

export function usePrefetchComparison(): (name: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (name: string) => {
      void prefetchApiQuery(qc, comparisonConfig(name));
    },
    [qc],
  );
}

export function comparisonPlotDownloadUrl(name: string, plotName: string): string {
  return fillPath(API_PATHS.comparisonPlot, { name, plot_name: plotName });
}
