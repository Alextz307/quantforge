import { http, HttpResponse } from "msw";
import { ROLE_ADMIN, ROLE_USER, type UserCreate, type UserPublic } from "@/api/users";

export const ADMIN_USER: UserPublic = { id: 1, username: "alex", role: ROLE_ADMIN };
export const REGULAR_USER: UserPublic = { id: 2, username: "guest", role: ROLE_USER };

export const SEED_USERS: UserPublic[] = [ADMIN_USER, REGULAR_USER];

export const handlers = [
  http.get("/api/auth/me", () => HttpResponse.json(ADMIN_USER)),
  http.post("/api/auth/login", () => HttpResponse.json(ADMIN_USER)),
  http.post("/api/auth/logout", () => new HttpResponse(null, { status: 204 })),
  http.get("/api/users", () => HttpResponse.json(SEED_USERS)),
  http.post("/api/users", async ({ request }) => {
    const body = (await request.json()) as UserCreate;
    return HttpResponse.json({ id: 99, username: body.username, role: body.role });
  }),
  http.delete("/api/users/:id", () => new HttpResponse(null, { status: 204 })),
];
