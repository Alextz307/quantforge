import {
  useMutation,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import type { ValidateResponse } from "./configs";
import { extractApiError } from "./errors";
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";
import { SaveUploadError } from "./studyUploads";

export { SaveUploadError } from "./studyUploads";

export type UniverseSpecUploadSummary = components["schemas"]["UniverseSpecUploadSummary"];
export type UniverseSpecUploadDetail = components["schemas"]["UniverseSpecUploadDetail"];
export type UniverseSpecUploadCreate = components["schemas"]["UniverseSpecUploadCreate"];
export type UniverseSpecValidateRequest = components["schemas"]["UniverseSpecValidateRequest"];

const LIST_STALE_TIME = 30_000;
const SCHEMA_STALE_TIME = Number.POSITIVE_INFINITY;

function uploadsConfig(): ApiQueryOptions<UniverseSpecUploadSummary[]> {
  return {
    queryKey: queryKeys.universeUploads,
    fetcher: () => apiClient.GET(API_PATHS.universeUploads),
    errorMsg: "Failed to load universe spec uploads",
    staleTime: LIST_STALE_TIME,
  };
}

function uploadDetailConfig(slug: string): ApiQueryOptions<UniverseSpecUploadDetail> {
  return {
    queryKey: queryKeys.universeUpload(slug),
    fetcher: () => apiClient.GET(API_PATHS.universeUpload, { params: { path: { slug } } }),
    errorMsg: "Failed to load universe spec upload",
    staleTime: LIST_STALE_TIME,
  };
}

function universeSpecSchemaConfig(): ApiQueryOptions<Record<string, unknown>> {
  return {
    queryKey: queryKeys.universeSpecSchema,
    fetcher: () => apiClient.GET(API_PATHS.universeSpecSchema),
    errorMsg: "Failed to load universe spec JSON schema",
    staleTime: SCHEMA_STALE_TIME,
    gcTime: SCHEMA_STALE_TIME,
  };
}

export function useUniverseUploads(): UseQueryResult<UniverseSpecUploadSummary[]> {
  return useApiQuery(uploadsConfig());
}

export function useUniverseUpload(slug: string | null): UseQueryResult<UniverseSpecUploadDetail> {
  const config = uploadDetailConfig(slug ?? "");
  return useApiQuery({ ...config, enabled: slug !== null });
}

export function useUniverseSpecSchema(): UseQueryResult<Record<string, unknown>> {
  return useApiQuery(universeSpecSchemaConfig());
}

export type ValidateUniverseSpecMutation = UseMutationResult<
  ValidateResponse,
  Error,
  UniverseSpecValidateRequest
>;

export function useValidateUniverseSpec(): ValidateUniverseSpecMutation {
  return useMutation({
    mutationFn: async (body: UniverseSpecValidateRequest): Promise<ValidateResponse> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.universeSpecValidate, {
        body,
      });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to validate universe spec"));
      }
      return data;
    },
  });
}

export type SaveUniverseUploadMutation = UseMutationResult<
  UniverseSpecUploadDetail,
  Error,
  UniverseSpecUploadCreate
>;

export function useSaveUniverseUpload(): SaveUniverseUploadMutation {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: UniverseSpecUploadCreate): Promise<UniverseSpecUploadDetail> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.universeUploads, {
        body,
      });
      if (response.ok && data) return data;
      throw new SaveUploadError(error, response.status);
    },
    onSuccess: (detail) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.universeUploads });
      queryClient.setQueryData(queryKeys.universeUpload(detail.slug), detail);
    },
  });
}

export type DeleteUniverseUploadMutation = UseMutationResult<void, Error, string>;

export function useDeleteUniverseUpload(): DeleteUniverseUploadMutation {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (slug: string): Promise<void> => {
      const { error, response } = await apiClient.DELETE(API_PATHS.universeUpload, {
        params: { path: { slug } },
      });
      if (!response.ok) {
        throw new Error(extractApiError(error, "Failed to delete universe spec upload"));
      }
    },
    onSuccess: (_void, slug) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.universeUploads });
      queryClient.removeQueries({ queryKey: queryKeys.universeUpload(slug) });
    },
  });
}
