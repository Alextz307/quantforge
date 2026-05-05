import { useQuery } from "@tanstack/react-query";
import { apiClient, type components } from "./client";
import { extractApiError } from "./errors";
import { API_PATHS, fillPath } from "./paths";
import { queryKeys } from "./queryKeys";

export type HoldoutEvalSummary = components["schemas"]["HoldoutEvalSummary"];
export type HoldoutEvalDetail = components["schemas"]["HoldoutEvalDetail"];

export function useHoldoutEvals() {
  return useQuery({
    queryKey: queryKeys.holdoutEvals,
    queryFn: async (): Promise<HoldoutEvalSummary[]> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.holdoutEvals);
      if (!response.ok || !data)
        throw new Error(extractApiError(error, "Failed to load holdout evaluations"));
      return data;
    },
    staleTime: 30_000,
  });
}

export function useHoldoutEval(name: string) {
  return useQuery({
    queryKey: queryKeys.holdoutEval(name),
    queryFn: async (): Promise<HoldoutEvalDetail> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.holdoutEval, {
        params: { path: { name } },
      });
      if (!response.ok || !data)
        throw new Error(extractApiError(error, "Failed to load holdout evaluation"));
      return data;
    },
    staleTime: Infinity,
  });
}

export function holdoutPlotDownloadUrl(name: string, plotName: string): string {
  return fillPath(API_PATHS.holdoutEvalPlot, { name, plot_name: plotName });
}
