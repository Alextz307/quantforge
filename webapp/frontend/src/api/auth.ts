import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { apiClient, type components } from "./client";
import { extractApiError } from "./errors";
import { queryKeys } from "./queryKeys";
import type { UserPublic } from "./users";
import { ROUTES } from "@/lib/routes";

export type LoginRequest = components["schemas"]["LoginRequest"];

const ME_PATH = "/api/auth/me";
const LOGIN_PATH = "/api/auth/login";
const LOGOUT_PATH = "/api/auth/logout";

export function useMe() {
  return useQuery({
    queryKey: queryKeys.me,
    queryFn: async (): Promise<UserPublic | null> => {
      const { data, error, response } = await apiClient.GET(ME_PATH);
      if (!response.ok) throw new Error(extractApiError(error, "Failed to fetch session"));
      return data ?? null;
    },
    retry: false,
    staleTime: 60_000,
  });
}

export function useLogin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: LoginRequest): Promise<UserPublic> => {
      const { data, error, response } = await apiClient.POST(LOGIN_PATH, { body });
      if (!response.ok || !data) {
        throw new Error(extractApiError(error, "Invalid username or password"));
      }
      return data;
    },
    onSuccess: (user) => {
      queryClient.setQueryData(queryKeys.me, user);
    },
  });
}

export function useLogout() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  return useMutation({
    mutationFn: async (): Promise<void> => {
      await apiClient.POST(LOGOUT_PATH);
    },
    onSuccess: () => {
      queryClient.removeQueries();
      navigate(ROUTES.login, { replace: true });
    },
  });
}
