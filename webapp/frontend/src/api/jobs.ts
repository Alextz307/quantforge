import { useMutation, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import { extractApiError, extractValidationItems } from "./errors";
import { API_PATHS, fillPath, wsUrlFor } from "./paths";
import { queryKeys } from "./queryKeys";

export type JobRow = components["schemas"]["JobRow"];
export type JobKind = components["schemas"]["JobKind"];
export type JobStatus = components["schemas"]["JobStatus"];
export type JobSubmission = components["schemas"]["JobSubmission"];
export type ValidationErrorItem = components["schemas"]["ValidationErrorItem"];

const TERMINAL_STATUSES: ReadonlySet<JobStatus> = new Set(["completed", "failed", "cancelled"]);

export function isTerminalStatus(status: JobStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

export interface JobsListOptions {
  allUsers?: boolean;
}

const JOBS_LIST_REFETCH_INTERVAL_MS = 5_000;

function jobsConfig(opts: JobsListOptions): ApiQueryOptions<JobRow[]> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.jobs({ allUsers }),
    fetcher: () =>
      allUsers
        ? apiClient.GET(API_PATHS.jobs, { params: { query: { all: true } } })
        : apiClient.GET(API_PATHS.jobs),
    errorMsg: "Failed to load jobs",
  };
}

function jobConfig(jobId: string): ApiQueryOptions<JobRow> {
  return {
    queryKey: queryKeys.job(jobId),
    fetcher: () => apiClient.GET(API_PATHS.job, { params: { path: { job_id: jobId } } }),
    errorMsg: "Failed to load job",
  };
}

export function useJobs(opts: JobsListOptions = {}): UseQueryResult<JobRow[]> {
  return useApiQuery<JobRow[]>({
    ...jobsConfig(opts),
    // Smart polling: keep cache fresh while at least one job is non-terminal,
    // pause otherwise. The hook itself is the change-detection guard so
    // pages don't have to wire their own intervals.
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data || data.length === 0) return false;
      return data.some((j) => !isTerminalStatus(j.status)) ? JOBS_LIST_REFETCH_INTERVAL_MS : false;
    },
  });
}

export function useJob(jobId: string): UseQueryResult<JobRow> {
  return useApiQuery(jobConfig(jobId));
}

export function useSubmitJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: JobSubmission): Promise<JobRow> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.jobs, { body });
      if (!response.ok || !data) {
        throw new SubmitJobError(error, response.status);
      }
      return data;
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.jobsAll });
    },
  });
}

export function useCancelJob(jobId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (): Promise<JobRow> => {
      const { data, error, response } = await apiClient.DELETE(API_PATHS.job, {
        params: { path: { job_id: jobId } },
      });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to cancel job"));
      }
      return data;
    },
    onSuccess: (job) => {
      qc.setQueryData(queryKeys.job(job.id), job);
      void qc.invalidateQueries({ queryKey: queryKeys.jobsAll });
    },
  });
}

/**
 * Carries the structured Pydantic errors when ``POST /api/jobs`` fails with
 * 422 so the form can surface inline messages keyed by ``loc``. Plain
 * ``Error`` discards the structured detail.
 */
export class SubmitJobError extends Error {
  readonly status: number;
  readonly fieldErrors: readonly ValidationErrorItem[];
  constructor(error: unknown, status: number) {
    // Pydantic 422 always emits ``ValidationErrorItem`` (loc + msg + type);
    // the runtime guard checks loc + msg only, so we trust the schema for ``type``.
    const fieldErrors = extractValidationItems(error) as readonly ValidationErrorItem[];
    const message =
      fieldErrors.length > 0
        ? fieldErrors.map((e) => `${e.loc.join(".")}: ${e.msg}`).join("; ")
        : extractApiError(error, "Failed to submit job");
    super(message);
    this.name = "SubmitJobError";
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

export function jobLogDownloadUrl(jobId: string): string {
  return fillPath(API_PATHS.jobLog, { job_id: jobId });
}

export function jobStreamUrl(jobId: string): string {
  return wsUrlFor(API_PATHS.jobStream, { job_id: jobId });
}
