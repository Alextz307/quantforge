import { act, render, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider, useTheme } from "@/lib/theme";
import { THEME_STORAGE_KEY } from "@/lib/themeStorage";

type MqlListener = (event: MediaQueryListEvent) => void;

interface FakeMql {
  matches: boolean;
  media: string;
  onchange: null;
  addListener: () => void;
  removeListener: () => void;
  addEventListener: (event: string, listener: MqlListener) => void;
  removeEventListener: (event: string, listener: MqlListener) => void;
  dispatchEvent: () => boolean;
  __listeners: MqlListener[];
}

function makeMatchMedia(initialDark: boolean) {
  let currentDark = initialDark;
  const mqls: FakeMql[] = [];
  const factory = (query: string): FakeMql => {
    const mql: FakeMql = {
      matches: query.includes("dark") ? currentDark : !currentDark,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: (_event: string, listener: MqlListener) => {
        mql.__listeners.push(listener);
      },
      removeEventListener: (_event: string, listener: MqlListener) => {
        mql.__listeners = mql.__listeners.filter((l) => l !== listener);
      },
      dispatchEvent: () => true,
      __listeners: [],
    };
    mqls.push(mql);
    return mql;
  };
  return {
    factory,
    setSystemDark(next: boolean) {
      currentDark = next;
      for (const mql of mqls) {
        mql.matches = mql.media.includes("dark") ? next : !next;
        for (const listener of mql.__listeners) {
          listener({ matches: mql.matches } as MediaQueryListEvent);
        }
      }
    },
  };
}

function Wrapper({ children }: { children: ReactNode }) {
  return <ThemeProvider>{children}</ThemeProvider>;
}

describe("ThemeProvider", () => {
  beforeEach(() => {
    document.documentElement.classList.remove("dark");
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("follows OS preference when nothing is stored", () => {
    const mm = makeMatchMedia(true);
    vi.stubGlobal("matchMedia", mm.factory);
    const { result } = renderHook(() => useTheme(), { wrapper: Wrapper });
    expect(result.current.resolvedTheme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("honors stored preference over OS preference", () => {
    const mm = makeMatchMedia(true);
    vi.stubGlobal("matchMedia", mm.factory);
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    const { result } = renderHook(() => useTheme(), { wrapper: Wrapper });
    expect(result.current.resolvedTheme).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("persists setTheme to localStorage and toggles the html class", () => {
    const mm = makeMatchMedia(false);
    vi.stubGlobal("matchMedia", mm.factory);
    const { result } = renderHook(() => useTheme(), { wrapper: Wrapper });
    act(() => {
      result.current.setTheme("dark");
    });
    expect(result.current.resolvedTheme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    act(() => {
      result.current.setTheme("light");
    });
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
  });

  it("tracks OS preference changes only until the user picks a theme", () => {
    const mm = makeMatchMedia(false);
    vi.stubGlobal("matchMedia", mm.factory);
    const { result } = renderHook(() => useTheme(), { wrapper: Wrapper });
    expect(result.current.resolvedTheme).toBe("light");
    act(() => {
      mm.setSystemDark(true);
    });
    expect(result.current.resolvedTheme).toBe("dark");
    act(() => {
      result.current.setTheme("light");
    });
    act(() => {
      mm.setSystemDark(false);
    });
    expect(result.current.resolvedTheme).toBe("light");
    act(() => {
      mm.setSystemDark(true);
    });
    expect(result.current.resolvedTheme).toBe("light");
  });

  it("throws when useTheme is called outside the provider", () => {
    const original = console.error;
    console.error = () => undefined;
    try {
      expect(() => render(<UseThemeProbe />)).toThrow(/ThemeProvider/);
    } finally {
      console.error = original;
    }
  });
});

function UseThemeProbe() {
  useTheme();
  return null;
}
