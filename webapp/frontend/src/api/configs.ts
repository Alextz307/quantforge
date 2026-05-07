import { useMutation, type UseMutationResult, type UseQueryResult } from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import { extractApiError } from "./errors";
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";

export type ConfigKind = components["schemas"]["ConfigKind"];
export type ConfigEntry = components["schemas"]["ConfigEntry"];
export type ConfigDetail = components["schemas"]["ConfigDetail"];
export type ValidateRequest = components["schemas"]["ValidateRequest"];
export type ValidateResponse = components["schemas"]["ValidateResponse"];
export type ValidationErrorItem = components["schemas"]["ValidationErrorItem"];

const LIST_STALE_TIME = 60_000;

function configsConfig(kind: ConfigKind): ApiQueryOptions<ConfigEntry[]> {
  return {
    queryKey: queryKeys.configs(kind),
    fetcher: () => apiClient.GET(API_PATHS.configs, { params: { path: { kind } } }),
    errorMsg: `Failed to load ${kind} configs`,
    staleTime: LIST_STALE_TIME,
  };
}

function configDetailConfig(kind: ConfigKind, name: string): ApiQueryOptions<ConfigDetail> {
  return {
    queryKey: queryKeys.configDetail(kind, name),
    fetcher: () => apiClient.GET(API_PATHS.configDetail, { params: { path: { kind, name } } }),
    errorMsg: "Failed to load config",
    staleTime: LIST_STALE_TIME,
  };
}

export function useConfigList(kind: ConfigKind): UseQueryResult<ConfigEntry[]> {
  return useApiQuery(configsConfig(kind));
}

export function useConfigDetail(
  kind: ConfigKind,
  name: string | null,
): UseQueryResult<ConfigDetail> {
  // ``name === null`` is the "nothing selected yet" state for a dependent
  // fetch (universe picker hasn't been touched). Disabled query keeps the
  // hook call-site stable while we wait on the user.
  const config = configDetailConfig(kind, name ?? "");
  return useApiQuery({ ...config, enabled: name !== null });
}

export type ValidateMutation = UseMutationResult<ValidateResponse, Error, ValidateRequest>;

export function useValidateConfig(): ValidateMutation {
  return useMutation({
    mutationFn: async (body: ValidateRequest): Promise<ValidateResponse> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.configValidate, { body });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to validate config"));
      }
      return data;
    },
  });
}
