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
import { queryKeys, type ComparisonsPageParams } from "./queryKeys";

export type ComparisonSummary = components["schemas"]["ComparisonSummary"];
export type ComparisonDetail = components["schemas"]["ComparisonDetail"];
export type ComparisonsPage = components["schemas"]["ComparisonsPage"];
export type PerStrategyStatsRow = components["schemas"]["PerStrategyStatsRow"];

const LIST_STALE_TIME = 30_000;
// Cap cache retention; paging/sort/filter combos spawn many short-lived keys.
const LIST_GC_TIME = 60_000;

export interface ComparisonsListOptions {
  allUsers?: boolean;
}

function comparisonsPageConfig(
  params: ComparisonsPageParams,
  opts: ComparisonsListOptions,
): ApiQueryOptions<ComparisonsPage> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.comparisonsPage({ ...params, allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.comparisons, {
        params: {
          query: {
            limit: params.limit,
            offset: params.offset,
            ...(params.strategy !== undefined ? { strategy: params.strategy } : {}),
            ...(params.since !== undefined ? { since: params.since } : {}),
            ...(allUsers ? { all: true } : {}),
          },
        },
      }),
    errorMsg: "Failed to load comparisons",
    staleTime: LIST_STALE_TIME,
    gcTime: LIST_GC_TIME,
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

export function useComparisonsPage(
  params: ComparisonsPageParams,
  opts: ComparisonsListOptions = {},
): UseQueryResult<ComparisonsPage> {
  return useApiQuery(comparisonsPageConfig(params, opts));
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
