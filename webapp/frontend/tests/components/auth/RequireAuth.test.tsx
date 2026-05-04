import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { ROUTES } from "@/lib/routes";
import { server } from "../../msw/server";
import { renderWithProviders } from "../../util/render";

const PROTECTED_TEXT = "secret content";
const LOGIN_TEXT = "login page";

function Tree() {
  return (
    <Routes>
      <Route path={ROUTES.login} element={<div>{LOGIN_TEXT}</div>} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <div>{PROTECTED_TEXT}</div>
          </RequireAuth>
        }
      />
    </Routes>
  );
}

describe("RequireAuth", () => {
  it("renders children when /api/auth/me succeeds", async () => {
    renderWithProviders(<Tree />);

    expect(await screen.findByText(PROTECTED_TEXT)).toBeInTheDocument();
  });

  it("redirects to login when /api/auth/me returns null", async () => {
    server.use(http.get("/api/auth/me", () => HttpResponse.json(null)));

    renderWithProviders(<Tree />);

    await waitFor(() => {
      expect(screen.getByText(LOGIN_TEXT)).toBeInTheDocument();
    });
    expect(screen.queryByText(PROTECTED_TEXT)).not.toBeInTheDocument();
  });
});
