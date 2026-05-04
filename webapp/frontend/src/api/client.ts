import createClient, { type Middleware } from "openapi-fetch";
import { FROM_QUERY_PARAM, ROUTES } from "@/lib/routes";
import type { paths } from "./generated/schema";

const UNAUTHORIZED = 401;
const AUTH_PATH_PREFIX = "/api/auth/";

const redirectOn401: Middleware = {
  onResponse({ response, request }) {
    if (response.status !== UNAUTHORIZED) return;
    if (typeof window === "undefined") return;
    if (window.location.pathname === ROUTES.login) return;
    if (new URL(request.url).pathname.startsWith(AUTH_PATH_PREFIX)) return;
    const from = window.location.pathname + window.location.search;
    const target = `${ROUTES.login}?${FROM_QUERY_PARAM}=${encodeURIComponent(from)}`;
    window.location.assign(target);
  },
};

const baseUrl = typeof window === "undefined" ? "http://localhost" : window.location.origin;

export const apiClient = createClient<paths>({
  baseUrl,
  credentials: "include",
  fetch: (...args) => globalThis.fetch(...args),
});

apiClient.use(redirectOn401);

export type { paths } from "./generated/schema";
export type { components } from "./generated/schema";
