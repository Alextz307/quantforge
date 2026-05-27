import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UserCreate } from "@/api/users";
import { UserList } from "@/features/admin/UserList";
import { ADMIN_USER, REGULAR_USER } from "../../msw/handlers";
import { server } from "../../msw/server";
import { renderWithProviders } from "../../util/render";

const NEW_USERNAME = "newbie";
const NEW_PASSWORD = "abcdefgh";

describe("UserList", () => {
  // Delete now goes through ``window.confirm`` — auto-accept so the existing
  // tests still exercise the request flow. The added confirmation-cancel test
  // overrides this per-case.
  beforeEach(() => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the seeded users", async () => {
    renderWithProviders(<UserList />);

    expect(await screen.findByText(ADMIN_USER.username)).toBeInTheDocument();
    expect(screen.getByText(REGULAR_USER.username)).toBeInTheDocument();
  });

  it("creates a user via the form", async () => {
    let createdBody: UserCreate | null = null;
    server.use(
      http.post("/api/users", async ({ request }) => {
        createdBody = (await request.json()) as UserCreate;
        return HttpResponse.json({
          id: 99,
          username: createdBody.username,
          role: createdBody.role,
        });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<UserList />);

    await screen.findByText(ADMIN_USER.username);
    await user.type(screen.getByLabelText(/Username/i), NEW_USERNAME);
    await user.type(screen.getByLabelText(/Password/i), NEW_PASSWORD);
    await user.click(screen.getByRole("button", { name: /Create/i }));

    await waitFor(() => {
      expect(createdBody).toEqual({
        username: NEW_USERNAME,
        password: NEW_PASSWORD,
        role: "user",
      });
    });
  });

  it("deletes a user when the delete button is clicked", async () => {
    let deletedId: string | null = null;
    server.use(
      http.delete("/api/users/:id", ({ params }) => {
        deletedId = String(params.id);
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<UserList />);

    const deleteBtn = await screen.findByRole("button", {
      name: new RegExp(`Delete ${ADMIN_USER.username}`, "i"),
    });
    await user.click(deleteBtn);

    await waitFor(() => {
      expect(deletedId).toBe(String(ADMIN_USER.id));
    });
  });

  it("surfaces a backend detail message when delete fails", async () => {
    server.use(
      http.delete("/api/users/:id", () =>
        HttpResponse.json({ detail: "Cannot delete the last admin" }, { status: 409 }),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<UserList />);

    await user.click(
      await screen.findByRole("button", {
        name: new RegExp(`Delete ${ADMIN_USER.username}`, "i"),
      }),
    );

    expect(await screen.findByText(/Cannot delete the last admin/i)).toBeInTheDocument();
  });

  it("surfaces a backend detail message when the user list fails to load", async () => {
    server.use(
      http.get("/api/users", () =>
        HttpResponse.json({ detail: "Database unavailable" }, { status: 503 }),
      ),
    );
    renderWithProviders(<UserList />);

    expect(await screen.findByText(/Database unavailable/i)).toBeInTheDocument();
  });

  it("does not send the DELETE request when the confirm dialog is cancelled", async () => {
    let deleteCalled = false;
    server.use(
      http.delete("/api/users/:id", () => {
        deleteCalled = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    renderWithProviders(<UserList />);

    await user.click(
      await screen.findByRole("button", {
        name: new RegExp(`Delete ${ADMIN_USER.username}`, "i"),
      }),
    );

    // Tiny settle so any pending mutate would have fired.
    await new Promise((r) => setTimeout(r, 50));
    expect(deleteCalled).toBe(false);
  });

  it("filters to auto-created users only when the toggle is enabled", async () => {
    const autoCreatedAt = "2026-05-27T12:00:00+00:00";
    server.use(
      http.get("/api/users", () =>
        HttpResponse.json([
          { id: 1, username: "alex", role: "admin", auto_created_at: null },
          { id: 2, username: "alxe", role: "user", auto_created_at: autoCreatedAt },
        ]),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<UserList />);

    expect(await screen.findByText("alex")).toBeInTheDocument();
    expect(screen.getByText("alxe")).toBeInTheDocument();

    await user.click(screen.getByTestId("users-auto-created-toggle"));

    await waitFor(() => {
      expect(screen.queryByText("alex")).not.toBeInTheDocument();
    });
    expect(screen.getByText("alxe")).toBeInTheDocument();
    expect(screen.getByText(autoCreatedAt)).toBeInTheDocument();
  });

  it("surfaces backend validation errors in the alert", async () => {
    server.use(
      http.post("/api/users", () =>
        HttpResponse.json(
          {
            detail: [
              {
                loc: ["body", "password"],
                msg: "String should have at least 8 characters",
                type: "string_too_short",
              },
            ],
          },
          { status: 422 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<UserList />);

    await screen.findByText(ADMIN_USER.username);
    await user.type(screen.getByLabelText(/Username/i), NEW_USERNAME);
    await user.type(screen.getByLabelText(/Password/i), NEW_PASSWORD);
    await user.click(screen.getByRole("button", { name: /Create/i }));

    expect(
      await screen.findByText(/password: String should have at least 8 characters/i),
    ).toBeInTheDocument();
  });
});
