import { useCallback } from "react";
import { useMutation, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  apiClient,
  prefetchApiQuery,
  useApiQuery,
  type ApiQueryOptions,
  type components,
} from "./client";
import { extractApiError } from "./errors";
import { API_PATHS, fillPath, wsUrlFor } from "./paths";
import { queryKeys } from "./queryKeys";

export type StudySummary = components["schemas"]["StudySummary"];
export type StudyDetail = components["schemas"]["StudyDetail"];
export type LegStateRow = components["schemas"]["LegStateRow"];
export type StudyConsolidatedDTO = components["schemas"]["StudyConsolidatedDTO"];

const LIST_STALE_TIME = 30_000;
const STUDY_DETAIL_STALE_TIME = 10_000;

export interface StudiesListOptions {
  allUsers?: boolean;
}

function studiesConfig(opts: StudiesListOptions): ApiQueryOptions<StudySummary[]> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.studiesList(allUsers),
    fetcher: () =>
      allUsers
        ? apiClient.GET(API_PATHS.studies, { params: { query: { all: true } } })
        : apiClient.GET(API_PATHS.studies),
    errorMsg: "Failed to load studies",
    staleTime: LIST_STALE_TIME,
  };
}

function studyConfig(name: string): ApiQueryOptions<StudyDetail> {
  return {
    queryKey: queryKeys.study(name),
    fetcher: () => apiClient.GET(API_PATHS.study, { params: { path: { name } } }),
    errorMsg: "Failed to load study",
    staleTime: STUDY_DETAIL_STALE_TIME,
  };
}

function studyConsolidatedConfig(name: string): ApiQueryOptions<StudyConsolidatedDTO> {
  return {
    queryKey: queryKeys.studyConsolidated(name),
    fetcher: () => apiClient.GET(API_PATHS.studyConsolidated, { params: { path: { name } } }),
    errorMsg: "Failed to load consolidated report",
    staleTime: Infinity,
  };
}

export function useStudies(opts: StudiesListOptions = {}): UseQueryResult<StudySummary[]> {
  return useApiQuery(studiesConfig(opts));
}

export function useStudy(name: string): UseQueryResult<StudyDetail> {
  return useApiQuery(studyConfig(name));
}

export function useStudyConsolidated(
  name: string,
  { enabled = true }: { enabled?: boolean } = {},
): UseQueryResult<StudyConsolidatedDTO> {
  return useApiQuery({ ...studyConsolidatedConfig(name), enabled });
}

export function useGenerateStudyConsolidated(name: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (): Promise<StudyConsolidatedDTO> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.studyConsolidated, {
        params: { path: { name } },
      });
      if (!response.ok || !data)
        throw new Error(extractApiError(error, "Failed to generate consolidated report"));
      return data;
    },
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.studyConsolidated(name), data);
      // Refresh useStudy so ``has_consolidated_report`` flips from false to true
      // and the page swaps the Generate-button branch for the report panel.
      void queryClient.invalidateQueries({ queryKey: queryKeys.study(name) });
    },
  });
}

export function usePrefetchStudy(): (name: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (name: string) => {
      void prefetchApiQuery(qc, studyConfig(name));
    },
    [qc],
  );
}

export function studyConsolidatedPlotUrl(name: string, plotName: string): string {
  return fillPath(API_PATHS.studyConsolidatedPlot, { name, plot_name: plotName });
}

export function studyConsolidatedTableUrl(name: string, tableName: string): string {
  return fillPath(API_PATHS.studyConsolidatedTable, { name, table_name: tableName });
}

export function studyStreamUrl(name: string): string {
  return wsUrlFor(API_PATHS.studyStream, { name });
}
