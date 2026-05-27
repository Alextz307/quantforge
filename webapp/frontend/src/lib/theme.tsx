import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { THEME_STORAGE_KEY, type ResolvedTheme } from "@/lib/themeStorage";

interface ThemeContextValue {
  resolvedTheme: ResolvedTheme;
  setTheme: (next: ResolvedTheme) => void;
}

const THEME_VALUES: readonly ResolvedTheme[] = ["light", "dark"];

const ThemeContext = createContext<ThemeContextValue | null>(null);

function isResolvedTheme(v: string | null): v is ResolvedTheme {
  return v !== null && (THEME_VALUES as readonly string[]).includes(v);
}

function systemPrefersDark(): boolean {
  if (typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function readStoredTheme(): ResolvedTheme | null {
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isResolvedTheme(raw) ? raw : null;
  } catch {
    return null;
  }
}

function initialResolvedTheme(): ResolvedTheme {
  return readStoredTheme() ?? (systemPrefersDark() ? "dark" : "light");
}

function applyClass(resolved: ResolvedTheme): void {
  const root = document.documentElement;
  if (resolved === "dark") root.classList.add("dark");
  else root.classList.remove("dark");
}

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(initialResolvedTheme);

  useEffect(() => {
    applyClass(resolvedTheme);
  }, [resolvedTheme]);

  // Track OS changes only while the user hasn't picked yet. Once they click the
  // toggle, the stored value pins the theme and OS changes are ignored.
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => {
      if (readStoredTheme() !== null) return;
      setResolvedTheme(event.matches ? "dark" : "light");
    };
    mql.addEventListener("change", onChange);
    return () => {
      mql.removeEventListener("change", onChange);
    };
  }, []);

  const setTheme = useCallback((next: ResolvedTheme) => {
    setResolvedTheme(next);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // private mode / quota
    }
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({ resolvedTheme, setTheme }),
    [resolvedTheme, setTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (ctx === null) throw new Error("useTheme must be used inside <ThemeProvider>");
  return ctx;
}
