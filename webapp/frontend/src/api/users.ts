import { useMutation, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { apiClient, useApiQuery, type ApiQueryOptions, type components } from "./client";
import { extractApiError } from "./errors";
import { queryKeys } from "./queryKeys";

export type UserPublic = components["schemas"]["UserPublic"];
export type UserCreate = components["schemas"]["UserCreate"];
export type Role = components["schemas"]["Role"];

export const ROLES = ["user", "admin"] as const satisfies readonly Role[];
export const ROLE_USER: Role = "user";
export const ROLE_ADMIN: Role = "admin";

const USERS_PATH = "/api/users";
const USER_BY_ID_PATH = "/api/users/{user_id}";

const LIST_STALE_TIME = 30_000;

function usersConfig(): ApiQueryOptions<UserPublic[]> {
  return {
    queryKey: queryKeys.users,
    fetcher: () => apiClient.GET(USERS_PATH),
    errorMsg: "Failed to load users",
    staleTime: LIST_STALE_TIME,
  };
}

export function useUsers(): UseQueryResult<UserPublic[]> {
  return useApiQuery(usersConfig());
}

export function useCreateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: UserCreate): Promise<UserPublic> => {
      const { data, error, response } = await apiClient.POST(USERS_PATH, { body });
      if (!response.ok || !data) throw new Error(extractApiError(error, "Failed to create user"));
      return data;
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.users });
    },
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (userId: number): Promise<void> => {
      const { error, response } = await apiClient.DELETE(USER_BY_ID_PATH, {
        params: { path: { user_id: userId } },
      });
      if (!response.ok) throw new Error(extractApiError(error, "Failed to delete user"));
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.users });
    },
  });
}
