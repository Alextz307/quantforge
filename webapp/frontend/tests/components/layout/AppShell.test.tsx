import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { ROUTES } from "@/lib/routes";
import { ADMIN_USER, REGULAR_USER } from "../../msw/handlers";
import { server } from "../../msw/server";
import { renderWithProviders } from "../../util/render";

function Tree() {
  return (
    <Routes>
      <Route element={<AppShell user={ADMIN_USER} />}>
        <Route path={ROUTES.runs} element={<div>runs page</div>} />
      </Route>
      <Route path={ROUTES.login} element={<div>login page</div>} />
    </Routes>
  );
}

describe("AppShell", () => {
  it("renders sidebar, header username, and outlet content for admins", () => {
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.runs] });

    expect(screen.getByText("QuantForge")).toBeInTheDocument();
    expect(screen.getByText(ADMIN_USER.username)).toBeInTheDocument();
    expect(screen.getByText("admin")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Runs/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Admin/i })).toBeInTheDocument();
    expect(screen.getByText("runs page")).toBeInTheDocument();
  });

  it("hides Admin nav link for regular users", () => {
    function RegularTree() {
      return (
        <Routes>
          <Route element={<AppShell user={REGULAR_USER} />}>
            <Route path={ROUTES.runs} element={<div>runs page</div>} />
          </Route>
        </Routes>
      );
    }
    renderWithProviders(<RegularTree />, { initialEntries: [ROUTES.runs] });

    expect(screen.queryByRole("link", { name: /Admin/i })).not.toBeInTheDocument();
  });

  it("posts to /api/auth/logout and navigates to the login page", async () => {
    let logoutCalled = false;
    server.use(
      http.post("/api/auth/logout", () => {
        logoutCalled = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<Tree />, { initialEntries: [ROUTES.runs] });

    await user.click(screen.getByRole("button", { name: /Sign out/i }));

    await waitFor(() => {
      expect(logoutCalled).toBe(true);
    });
    expect(await screen.findByText("login page")).toBeInTheDocument();
  });
});
