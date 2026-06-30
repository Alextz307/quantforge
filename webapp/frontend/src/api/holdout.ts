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
import { API_PATHS, fillPath } from "./paths";
import { queryKeys, type HoldoutEvalsPageParams } from "./queryKeys";

export type HoldoutEvalSummary = components["schemas"]["HoldoutEvalSummary"];
export type HoldoutEvalDetail = components["schemas"]["HoldoutEvalDetail"];
export type HoldoutEvalsPage = components["schemas"]["HoldoutEvalsPage"];
export type HoldoutSortBy = components["schemas"]["HoldoutSortBy"];

const LIST_STALE_TIME = 30_000;
// Cap cache retention; paging/sort/filter combos spawn many short-lived keys.
const LIST_GC_TIME = 60_000;

export interface HoldoutEvalsListOptions {
  allUsers?: boolean;
}

function holdoutEvalsConfig(
  opts: HoldoutEvalsListOptions,
): ApiQueryOptions<HoldoutEvalsPage, HoldoutEvalSummary[]> {
  // The deployment picker wants the whole visible set; request the max page.
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.holdoutEvalsPicker({ allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.holdoutEvals, {
        params: { query: { limit: MAX_PAGE_LIMIT, ...(allUsers ? { all: true } : {}) } },
      }),
    errorMsg: "Failed to load holdout evaluations",
    staleTime: LIST_STALE_TIME,
    select: (page) => page.items,
  };
}

function holdoutEvalsPageConfig(
  params: HoldoutEvalsPageParams,
  opts: HoldoutEvalsListOptions,
): ApiQueryOptions<HoldoutEvalsPage> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.holdoutEvalsPage({ ...params, allUsers }),
    fetcher: () =>
      apiClient.GET(API_PATHS.holdoutEvals, {
        params: {
          query: {
            limit: params.limit,
            offset: params.offset,
            sort_by: params.sortBy,
            order: params.order,
            ...(params.sourceKind !== undefined ? { source_kind: params.sourceKind } : {}),
            ...(params.since !== undefined ? { since: params.since } : {}),
            ...(allUsers ? { all: true } : {}),
          },
        },
      }),
    errorMsg: "Failed to load holdout evaluations",
    staleTime: LIST_STALE_TIME,
    gcTime: LIST_GC_TIME,
  };
}

function holdoutEvalConfig(name: string): ApiQueryOptions<HoldoutEvalDetail> {
  return {
    queryKey: queryKeys.holdoutEval(name),
    fetcher: () => apiClient.GET(API_PATHS.holdoutEval, { params: { path: { name } } }),
    errorMsg: "Failed to load holdout evaluation",
    staleTime: Infinity,
  };
}

export function useHoldoutEvalsPage(
  params: HoldoutEvalsPageParams,
  opts: HoldoutEvalsListOptions = {},
): UseQueryResult<HoldoutEvalsPage> {
  return useApiQuery(holdoutEvalsPageConfig(params, opts));
}

export function useHoldoutEvals(
  opts: HoldoutEvalsListOptions = {},
): UseQueryResult<HoldoutEvalSummary[]> {
  return useApiQuery(holdoutEvalsConfig(opts));
}

export function useHoldoutEval(name: string): UseQueryResult<HoldoutEvalDetail> {
  return useApiQuery(holdoutEvalConfig(name));
}

export function usePrefetchHoldoutEval(): (name: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (name: string) => {
      void prefetchApiQuery(qc, holdoutEvalConfig(name));
    },
    [qc],
  );
}

export function holdoutPlotDownloadUrl(name: string, plotName: string): string {
  return fillPath(API_PATHS.holdoutEvalPlot, { name, plot_name: plotName });
}
