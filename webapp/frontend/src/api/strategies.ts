import { type UseQueryResult } from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";

export type RegistryEntry = components["schemas"]["RegistryEntry"];
export type StrategySchema = components["schemas"]["StrategySchema"];
export type StrategyParam = components["schemas"]["StrategyParam"];
export type ParamKind = components["schemas"]["ParamKind"];

const SCHEMA_STALE_TIME = 5 * 60_000;

function strategiesConfig(): ApiQueryOptions<RegistryEntry[]> {
  return {
    queryKey: queryKeys.strategies,
    fetcher: () => apiClient.GET(API_PATHS.strategies),
    errorMsg: "Failed to load strategies",
    staleTime: SCHEMA_STALE_TIME,
  };
}

function strategySchemaConfig(name: string): ApiQueryOptions<StrategySchema> {
  return {
    queryKey: queryKeys.strategySchema(name),
    fetcher: () => apiClient.GET(API_PATHS.strategySchema, { params: { path: { name } } }),
    errorMsg: "Failed to load strategy schema",
    staleTime: SCHEMA_STALE_TIME,
  };
}

export function useStrategies(): UseQueryResult<RegistryEntry[]> {
  return useApiQuery(strategiesConfig());
}

export function useStrategySchema(name: string | null): UseQueryResult<StrategySchema> {
  const config = strategySchemaConfig(name ?? "");
  return useApiQuery({ ...config, enabled: name !== null });
}
