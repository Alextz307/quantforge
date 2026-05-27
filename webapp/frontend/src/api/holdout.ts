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

export type HoldoutEvalSummary = components["schemas"]["HoldoutEvalSummary"];
export type HoldoutEvalDetail = components["schemas"]["HoldoutEvalDetail"];

const LIST_STALE_TIME = 30_000;

export interface HoldoutEvalsListOptions {
  allUsers?: boolean;
}

function holdoutEvalsConfig(
  opts: HoldoutEvalsListOptions,
): ApiQueryOptions<HoldoutEvalSummary[]> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.holdoutEvalsList(allUsers),
    fetcher: () =>
      allUsers
        ? apiClient.GET(API_PATHS.holdoutEvals, { params: { query: { all: true } } })
        : apiClient.GET(API_PATHS.holdoutEvals),
    errorMsg: "Failed to load holdout evaluations",
    staleTime: LIST_STALE_TIME,
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
