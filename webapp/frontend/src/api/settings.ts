import { type UseQueryResult } from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";

export type PublicSettings = components["schemas"]["PublicSettings"];

const SETTINGS_STALE_TIME = 60 * 60_000;

function publicSettingsConfig(): ApiQueryOptions<PublicSettings> {
  return {
    queryKey: queryKeys.publicSettings,
    fetcher: () => apiClient.GET(API_PATHS.publicSettings),
    errorMsg: "Failed to load public settings",
    staleTime: SETTINGS_STALE_TIME,
  };
}

export function usePublicSettings(): UseQueryResult<PublicSettings> {
  return useApiQuery(publicSettingsConfig());
}
