import {
  useMutation,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import { extractApiError } from "./errors";
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";
import type { ValidationErrorItem, ValidateResponse } from "./configs";

export type StudySpecUploadSummary = components["schemas"]["StudySpecUploadSummary"];
export type StudySpecUploadDetail = components["schemas"]["StudySpecUploadDetail"];
export type StudySpecUploadCreate = components["schemas"]["StudySpecUploadCreate"];
export type StudySpecValidateRequest = components["schemas"]["StudySpecValidateRequest"];

const LIST_STALE_TIME = 30_000;
// JSON Schema is generated from frozen Pydantic models - never changes within
// a server build, so we lean on a generous staleTime + gcTime to make hovers
// and per-keystroke validation cheap.
const SCHEMA_STALE_TIME = Number.POSITIVE_INFINITY;

function uploadsConfig(): ApiQueryOptions<StudySpecUploadSummary[]> {
  return {
    queryKey: queryKeys.studyUploads,
    fetcher: () => apiClient.GET(API_PATHS.studyUploads),
    errorMsg: "Failed to load study spec uploads",
    staleTime: LIST_STALE_TIME,
  };
}

function uploadDetailConfig(slug: string): ApiQueryOptions<StudySpecUploadDetail> {
  return {
    queryKey: queryKeys.studyUpload(slug),
    fetcher: () => apiClient.GET(API_PATHS.studyUpload, { params: { path: { slug } } }),
    errorMsg: "Failed to load study spec upload",
    staleTime: LIST_STALE_TIME,
  };
}

function studySpecSchemaConfig(): ApiQueryOptions<Record<string, unknown>> {
  return {
    queryKey: queryKeys.studySpecSchema,
    fetcher: () => apiClient.GET(API_PATHS.studySpecSchema),
    errorMsg: "Failed to load study spec JSON schema",
    staleTime: SCHEMA_STALE_TIME,
    gcTime: SCHEMA_STALE_TIME,
  };
}

export function useStudyUploads(): UseQueryResult<StudySpecUploadSummary[]> {
  return useApiQuery(uploadsConfig());
}

export function useStudyUpload(slug: string | null): UseQueryResult<StudySpecUploadDetail> {
  const config = uploadDetailConfig(slug ?? "");
  return useApiQuery({ ...config, enabled: slug !== null });
}

export function useStudySpecSchema(): UseQueryResult<Record<string, unknown>> {
  return useApiQuery(studySpecSchemaConfig());
}

export type ValidateStudySpecMutation = UseMutationResult<
  ValidateResponse,
  Error,
  StudySpecValidateRequest
>;

export function useValidateStudySpec(): ValidateStudySpecMutation {
  return useMutation({
    mutationFn: async (body: StudySpecValidateRequest): Promise<ValidateResponse> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.studySpecValidate, {
        body,
      });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to validate study spec"));
      }
      return data;
    },
  });
}

function deriveSaveUploadFieldErrors(
  error: unknown,
  status: number,
): { fieldErrors: readonly ValidationErrorItem[]; isLibraryCollision: boolean } {
  // FastAPI 422 -> detail is ValidationErrorItem[]. 409 -> string detail (slug
  // collision). Anything else -> flatten the message into a single slug-anchored
  // entry so <ServerErrorList> still has something to render.
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (status === 422 && Array.isArray(detail)) {
    return { fieldErrors: detail as readonly ValidationErrorItem[], isLibraryCollision: false };
  }
  if (status === 409) {
    const msg = typeof detail === "string" ? detail : "slug collides with library";
    return {
      fieldErrors: [{ loc: ["slug"], msg, type: "value_error" }],
      isLibraryCollision: true,
    };
  }
  return {
    fieldErrors: [
      {
        loc: ["yaml"],
        msg: extractApiError(error, "Failed to save study spec upload"),
        type: "value_error",
      },
    ],
    isLibraryCollision: false,
  };
}

export class SaveUploadError extends Error {
  readonly status: number;
  readonly fieldErrors: readonly ValidationErrorItem[];
  readonly isLibraryCollision: boolean;

  constructor(error: unknown, status: number) {
    const { fieldErrors, isLibraryCollision } = deriveSaveUploadFieldErrors(error, status);
    const message = fieldErrors.map((e) => `${e.loc.join(".")}: ${e.msg}`).join("; ");
    super(message);
    this.name = "SaveUploadError";
    this.status = status;
    this.fieldErrors = fieldErrors;
    this.isLibraryCollision = isLibraryCollision;
  }
}

export type SaveUploadMutation = UseMutationResult<
  StudySpecUploadDetail,
  Error,
  StudySpecUploadCreate
>;

export function useSaveStudyUpload(): SaveUploadMutation {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: StudySpecUploadCreate): Promise<StudySpecUploadDetail> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.studyUploads, {
        body,
      });
      if (response.ok && data) return data;
      throw new SaveUploadError(error, response.status);
    },
    onSuccess: (detail) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.studyUploads });
      queryClient.setQueryData(queryKeys.studyUpload(detail.slug), detail);
    },
  });
}

export type DeleteUploadMutation = UseMutationResult<void, Error, string>;

export function useDeleteStudyUpload(): DeleteUploadMutation {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (slug: string): Promise<void> => {
      const { error, response } = await apiClient.DELETE(API_PATHS.studyUpload, {
        params: { path: { slug } },
      });
      if (!response.ok) {
        throw new Error(extractApiError(error, "Failed to delete study spec upload"));
      }
    },
    onSuccess: (_void, slug) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.studyUploads });
      queryClient.removeQueries({ queryKey: queryKeys.studyUpload(slug) });
    },
  });
}
