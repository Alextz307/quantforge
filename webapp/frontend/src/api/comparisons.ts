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
import { queryKeys, type ComparisonsPageParams } from "./queryKeys";

export type ComparisonSummary = components["schemas"]["ComparisonSummary"];
export type ComparisonDetail = components["schemas"]["ComparisonDetail"];
export type ComparisonsPage = components["schemas"]["ComparisonsPage"];
export type PerStrategyStatsRow = components["schemas"]["PerStrategyStatsRow"];

export interface ComparisonsListOptions {
  allUsers?: boolean;
}

function comparisonsPageConfig(
  params: ComparisonsPageParams,
  opts: ComparisonsListOptions,
): ApiQueryOptions<ComparisonsPage> {
  const allUsers = opts.allUsers ?? false;
  return listPageConfig({
    queryKey: queryKeys.comparisonsPage({ ...params, allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.comparisons, {
        params: {
          query: {
            // openapi-fetch omits null/undefined query values; `?? null` keeps
            // exactOptionalPropertyTypes satisfied while dropping absent filters.
            limit: params.limit,
            offset: params.offset,
            strategy: params.strategy ?? null,
            since: params.since ?? null,
            ...(allUsers ? { all: true } : {}),
          },
        },
      }),
    errorMsg: "Failed to load comparisons",
  });
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
