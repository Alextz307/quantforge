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

export type RegimeReportSummary = components["schemas"]["RegimeReportSummary"];
export type RegimeReportDetail = components["schemas"]["RegimeReportDetail"];
export type PerRegimeStatsRow = components["schemas"]["PerRegimeStatsRow"];
export type RegimeSliceDTO = components["schemas"]["RegimeSliceDTO"];

const LIST_STALE_TIME = 30_000;

function regimeReportsConfig(): ApiQueryOptions<RegimeReportSummary[]> {
  return {
    queryKey: queryKeys.regimeReports,
    fetcher: () => apiClient.GET(API_PATHS.regimeReports),
    errorMsg: "Failed to load regime reports",
    staleTime: LIST_STALE_TIME,
  };
}

function regimeReportConfig(name: string): ApiQueryOptions<RegimeReportDetail> {
  return {
    queryKey: queryKeys.regimeReport(name),
    fetcher: () => apiClient.GET(API_PATHS.regimeReport, { params: { path: { name } } }),
    errorMsg: "Failed to load regime report",
    staleTime: Infinity,
  };
}

export function useRegimeReports(): UseQueryResult<RegimeReportSummary[]> {
  return useApiQuery(regimeReportsConfig());
}

export function useRegimeReport(name: string): UseQueryResult<RegimeReportDetail> {
  return useApiQuery(regimeReportConfig(name));
}

export function usePrefetchRegimeReport(): (name: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (name: string) => {
      void prefetchApiQuery(qc, regimeReportConfig(name));
    },
    [qc],
  );
}

export function regimePlotDownloadUrl(name: string, plotName: string): string {
  return fillPath(API_PATHS.regimePlot, { name, plot_name: plotName });
}
