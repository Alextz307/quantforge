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

export type StudySummary = components["schemas"]["StudySummary"];
export type StudyDetail = components["schemas"]["StudyDetail"];
export type LegStateRow = components["schemas"]["LegStateRow"];
export type StudyConsolidatedDTO = components["schemas"]["StudyConsolidatedDTO"];

const LIST_STALE_TIME = 30_000;
const STUDY_DETAIL_STALE_TIME = 10_000;

function studiesConfig(): ApiQueryOptions<StudySummary[]> {
  return {
    queryKey: queryKeys.studies,
    fetcher: () => apiClient.GET(API_PATHS.studies),
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

export function useStudies(): UseQueryResult<StudySummary[]> {
  return useApiQuery(studiesConfig());
}

export function useStudy(name: string): UseQueryResult<StudyDetail> {
  return useApiQuery(studyConfig(name));
}

export function useStudyConsolidated(name: string): UseQueryResult<StudyConsolidatedDTO> {
  return useApiQuery(studyConsolidatedConfig(name));
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
