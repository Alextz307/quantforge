import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider } from "@/lib/theme";
import { THEME_STORAGE_KEY } from "@/lib/themeStorage";
import { ThemeToggle } from "@/components/ThemeToggle";

function fakeMatchMedia(prefersDark = false) {
  const factory = (query: string) => ({
    matches: query.includes("dark") ? prefersDark : !prefersDark,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => true,
  });
  vi.stubGlobal("matchMedia", factory);
}

describe("ThemeToggle", () => {
  beforeEach(() => {
    document.documentElement.classList.remove("dark");
    window.localStorage.clear();
    fakeMatchMedia();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("toggles between light and dark and persists the choice", async () => {
    const user = userEvent.setup();
    render(
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>,
    );
    const toggle = screen.getByTestId("theme-toggle");
    expect(toggle).toHaveAttribute("data-theme", "light");
    expect(toggle).toHaveAttribute("aria-label", "Switch to dark theme");
    await user.click(toggle);
    expect(toggle).toHaveAttribute("data-theme", "dark");
    expect(toggle).toHaveAttribute("aria-label", "Switch to light theme");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    await user.click(toggle);
    expect(toggle).toHaveAttribute("data-theme", "light");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
  });
});
