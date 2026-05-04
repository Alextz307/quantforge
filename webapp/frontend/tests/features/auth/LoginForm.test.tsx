import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { LoginForm } from "@/features/auth/LoginForm";
import { server } from "../../msw/server";
import { renderWithProviders } from "../../util/render";

describe("LoginForm", () => {
  it("shows validation errors when fields are empty", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LoginForm />);

    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText(/Username is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/Password is required/i)).toBeInTheDocument();
  });

  it("submits credentials and lets the caller navigate on success", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LoginForm />);

    await user.type(screen.getByLabelText(/Username/i), "alex");
    await user.type(screen.getByLabelText(/Password/i), "secret-pw");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.queryByText(/Invalid username or password/i)).not.toBeInTheDocument();
    });
  });

  it("surfaces the backend detail message when login fails", async () => {
    server.use(
      http.post("/api/auth/login", () =>
        HttpResponse.json({ detail: "Invalid credentials" }, { status: 401 }),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<LoginForm />);

    await user.type(screen.getByLabelText(/Username/i), "alex");
    await user.type(screen.getByLabelText(/Password/i), "wrong-pw");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText(/Invalid credentials/i)).toBeInTheDocument();
  });
});
