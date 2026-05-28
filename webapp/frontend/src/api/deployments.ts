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
import { API_PATHS } from "./paths";
import { queryKeys } from "./queryKeys";

export type DeploymentSummary = components["schemas"]["DeploymentSummary"];
export type DeploymentDetail = components["schemas"]["DeploymentDetail"];
export type DeploymentCreate = components["schemas"]["DeploymentCreate"];
export type SignalRowOut = components["schemas"]["SignalRowOut"];
export type PredictIfStaleResponse = components["schemas"]["PredictIfStaleResponse"];

const LIST_STALE_TIME = 30_000;

export interface DeploymentsListOptions {
  allUsers?: boolean;
}

function deploymentsConfig(opts: DeploymentsListOptions): ApiQueryOptions<DeploymentSummary[]> {
  const allUsers = opts.allUsers ?? false;
  return {
    queryKey: queryKeys.deploymentsList(allUsers),
    fetcher: () =>
      allUsers
        ? apiClient.GET(API_PATHS.deployments, { params: { query: { all: true } } })
        : apiClient.GET(API_PATHS.deployments),
    errorMsg: "Failed to load deployments",
    staleTime: LIST_STALE_TIME,
  };
}

function deploymentConfig(id: string): ApiQueryOptions<DeploymentDetail> {
  return {
    queryKey: queryKeys.deployment(id),
    fetcher: () => apiClient.GET(API_PATHS.deployment, { params: { path: { deployment_id: id } } }),
    errorMsg: "Failed to load deployment",
    staleTime: Infinity,
  };
}

function deploymentSignalsConfig(id: string): ApiQueryOptions<SignalRowOut[]> {
  return {
    queryKey: queryKeys.deploymentSignals(id),
    fetcher: () =>
      apiClient.GET(API_PATHS.deploymentSignals, { params: { path: { deployment_id: id } } }),
    errorMsg: "Failed to load signal history",
    staleTime: Infinity,
  };
}

export function useDeployments(
  opts: DeploymentsListOptions = {},
): UseQueryResult<DeploymentSummary[]> {
  return useApiQuery(deploymentsConfig(opts));
}

export function useDeployment(id: string): UseQueryResult<DeploymentDetail> {
  return useApiQuery(deploymentConfig(id));
}

export function useDeploymentSignals(id: string): UseQueryResult<SignalRowOut[]> {
  return useApiQuery(deploymentSignalsConfig(id));
}

export function usePrefetchDeployment(): (id: string) => void {
  const qc = useQueryClient();
  return useCallback(
    (id: string) => {
      void prefetchApiQuery(qc, deploymentConfig(id));
    },
    [qc],
  );
}

export function useCreateDeployment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: DeploymentCreate): Promise<DeploymentDetail> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.deployments, { body });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to create deployment"));
      }
      return data;
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.deployments });
    },
  });
}

export function useRenameDeployment(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (name: string): Promise<DeploymentDetail> => {
      const { data, error, response } = await apiClient.PATCH(API_PATHS.deployment, {
        params: { path: { deployment_id: id } },
        body: { name },
      });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to rename deployment"));
      }
      return data;
    },
    onSuccess: (detail) => {
      qc.setQueryData(queryKeys.deployment(id), detail);
      void qc.invalidateQueries({ queryKey: queryKeys.deployments });
    },
  });
}

export function useDeleteDeployment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (id: string): Promise<void> => {
      const { error, response } = await apiClient.DELETE(API_PATHS.deployment, {
        params: { path: { deployment_id: id } },
      });
      if (!response.ok) {
        throw new Error(extractApiError(error, "Failed to delete deployment"));
      }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.deployments });
    },
  });
}

export function usePredictIfStale(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (): Promise<PredictIfStaleResponse> => {
      const { data, error, response } = await apiClient.POST(API_PATHS.deploymentPredict, {
        params: { path: { deployment_id: id } },
      });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Failed to compute signal"));
      }
      return data;
    },
    onSuccess: (res) => {
      // A fresh predict appended a row and moved latest_signal; a recall changed
      // nothing on disk, so only invalidate the cached views when stale.
      if (res.stale) {
        void qc.invalidateQueries({ queryKey: queryKeys.deployment(id) });
        void qc.invalidateQueries({ queryKey: queryKeys.deploymentSignals(id) });
      }
    },
  });
}
