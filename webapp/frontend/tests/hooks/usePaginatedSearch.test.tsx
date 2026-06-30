import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MAX_PAGE_LIMIT } from "@/api/client";
import { DEFAULT_PAGE_LIMIT, usePaginatedSearch } from "@/hooks/usePaginatedSearch";
import { ROUTER_FUTURE_FLAGS } from "../util/router";

const CUSTOM_LIMIT = 25;
const OVER_CAP_LIMIT = MAX_PAGE_LIMIT + 100;
const PAGE_TWO_OFFSET = 50;
const PAGE_THREE_OFFSET = 100;

function wrapperFor(initialEntry: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter initialEntries={[initialEntry]} future={ROUTER_FUTURE_FLAGS}>
        {children}
      </MemoryRouter>
    );
  };
}

describe("usePaginatedSearch", () => {
  it("defaults to the first page at the default limit", () => {
    const { result } = renderHook(() => usePaginatedSearch(), { wrapper: wrapperFor("/") });
    expect(result.current.limit).toBe(DEFAULT_PAGE_LIMIT);
    expect(result.current.offset).toBe(0);
  });

  it("reads limit and offset from the URL", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor(`/?limit=${String(CUSTOM_LIMIT)}&offset=${String(PAGE_TWO_OFFSET)}`),
    });
    expect(result.current.limit).toBe(CUSTOM_LIMIT);
    expect(result.current.offset).toBe(PAGE_TWO_OFFSET);
  });

  it("clamps an over-cap ?limit to the backend maximum", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor(`/?limit=${String(OVER_CAP_LIMIT)}`),
    });
    expect(result.current.limit).toBe(MAX_PAGE_LIMIT);
  });

  it("falls back to defaults for unparseable limit and negative offset", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor("/?limit=abc&offset=-3"),
    });
    expect(result.current.limit).toBe(DEFAULT_PAGE_LIMIT);
    expect(result.current.offset).toBe(0);
  });

  it("setOffset moves only the offset and preserves other params", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor("/?strategy=Foo"),
    });
    act(() => {
      result.current.setOffset(PAGE_THREE_OFFSET);
    });
    expect(result.current.offset).toBe(PAGE_THREE_OFFSET);
    expect(result.current.searchParams.get("strategy")).toBe("Foo");
  });

  it("setOffset clears the offset param when returning to the first page", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor(`/?offset=${String(PAGE_TWO_OFFSET)}`),
    });
    act(() => {
      result.current.setOffset(0);
    });
    expect(result.current.offset).toBe(0);
    expect(result.current.searchParams.has("offset")).toBe(false);
  });

  it("setParams applies updates and resets offset to the first page", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor(`/?offset=${String(PAGE_TWO_OFFSET)}`),
    });
    act(() => {
      result.current.setParams({ strategy: "Bar" });
    });
    expect(result.current.searchParams.get("strategy")).toBe("Bar");
    expect(result.current.offset).toBe(0);
  });

  it("setParams deletes a param given an empty value", () => {
    const { result } = renderHook(() => usePaginatedSearch(), {
      wrapper: wrapperFor("/?strategy=Foo"),
    });
    act(() => {
      result.current.setParams({ strategy: "" });
    });
    expect(result.current.searchParams.has("strategy")).toBe(false);
  });
});
