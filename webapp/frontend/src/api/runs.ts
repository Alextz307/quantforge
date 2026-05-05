import { useQuery } from "@tanstack/react-query";
import { apiClient, type components } from "./client";
import { extractApiError } from "./errors";
import { API_PATHS, fillPath } from "./paths";
import { queryKeys } from "./queryKeys";

export type RunSummary = components["schemas"]["RunSummary"];
export type RunDetail = components["schemas"]["RunDetail"];
export type FoldRow = components["schemas"]["FoldRow"];

export function useRuns() {
  return useQuery({
    queryKey: queryKeys.runs,
    queryFn: async (): Promise<RunSummary[]> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.runs);
      if (!response.ok || !data) throw new Error(extractApiError(error, "Failed to load runs"));
      return data;
    },
    staleTime: 30_000,
  });
}

export function useRun(experimentId: string) {
  return useQuery({
    queryKey: queryKeys.run(experimentId),
    queryFn: async (): Promise<RunDetail> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.run, {
        params: { path: { experiment_id: experimentId } },
      });
      if (!response.ok || !data) throw new Error(extractApiError(error, "Failed to load run"));
      return data;
    },
    staleTime: Infinity,
  });
}

export function useRunFolds(experimentId: string) {
  return useQuery({
    queryKey: queryKeys.runFolds(experimentId),
    queryFn: async (): Promise<FoldRow[]> => {
      const { data, error, response } = await apiClient.GET(API_PATHS.runFolds, {
        params: { path: { experiment_id: experimentId } },
      });
      if (!response.ok || !data) throw new Error(extractApiError(error, "Failed to load folds"));
      return data;
    },
    staleTime: Infinity,
  });
}

export function plotDownloadUrl(experimentId: string, plotName: string): string {
  return fillPath(API_PATHS.runPlot, { experiment_id: experimentId, plot_name: plotName });
}
