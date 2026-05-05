import { useQuery } from "@tanstack/react-query";
import { apiClient, type components } from "./client";
import { extractApiError } from "./errors";
import { API_PATHS, fillPath } from "./paths";
import { queryKeys } from "./queryKeys";

export type ComparisonSummary = components["schemas"]["ComparisonSummary"];
export type ComparisonDetail = components["schemas"]["ComparisonDetail"];
export type PerStrategyStatsRow = components["schemas"]["PerStrategyStatsRow"];

export function useComparisons() {
  return useQuery({
    queryKey: queryKeys.comparisons,
    queryFn: async (): Promise<ComparisonSummary[]> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.comparisons);
      if (!response.ok || !data)
        throw new Error(extractApiError(error, "Failed to load comparisons"));
      return data;
    },
    staleTime: 30_000,
  });
}

export function useComparison(name: string) {
  return useQuery({
    queryKey: queryKeys.comparison(name),
    queryFn: async (): Promise<ComparisonDetail> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.comparison, {
        params: { path: { name } },
      });
      if (!response.ok || !data)
        throw new Error(extractApiError(error, "Failed to load comparison"));
      return data;
    },
    staleTime: Infinity,
  });
}

export function comparisonPlotDownloadUrl(name: string, plotName: string): string {
  return fillPath(API_PATHS.comparisonPlot, { name, plot_name: plotName });
}
