import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { useMe } from "@/api/auth";
import { RequireAdmin } from "@/components/auth/RequireAdmin";
import { server } from "../../msw/server";
import { REGULAR_USER } from "../../msw/handlers";
import { renderWithProviders } from "../../util/render";

const PROTECTED_TEXT = "admin content";

function Tree() {
  const { data } = useMe();
  if (!data) return null;
  return (
    <RequireAdmin>
      <div>{PROTECTED_TEXT}</div>
    </RequireAdmin>
  );
}

describe("RequireAdmin", () => {
  it("renders children for admin users", async () => {
    renderWithProviders(<Tree />);

    expect(await screen.findByText(PROTECTED_TEXT)).toBeInTheDocument();
  });

  it("renders Admin only message for regular users", async () => {
    server.use(http.get("/api/auth/me", () => HttpResponse.json(REGULAR_USER)));

    renderWithProviders(<Tree />);

    await waitFor(() => {
      expect(screen.getByText(/Admin only/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(PROTECTED_TEXT)).not.toBeInTheDocument();
  });
});
