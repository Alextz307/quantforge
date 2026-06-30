import { useSearchParams } from "react-router-dom";
import { MAX_PAGE_LIMIT } from "@/api/client";

export const DEFAULT_PAGE_LIMIT = 50;

export interface PaginatedSearch {
  searchParams: URLSearchParams;
  limit: number;
  offset: number;
  setOffset: (offset: number) => void;
  // Filter/sort changes: apply the given params and jump back to the first page
  // so the user never lands on an out-of-range offset after narrowing the set.
  setParams: (updates: Record<string, string>) => void;
}

function readInt(raw: string | null, fallback: number, min: number, max?: number): number {
  const parsed = Number.parseInt(raw ?? "", 10);
  if (!Number.isFinite(parsed) || parsed < min) return fallback;
  return max !== undefined ? Math.min(parsed, max) : parsed;
}

export function usePaginatedSearch(defaultLimit = DEFAULT_PAGE_LIMIT): PaginatedSearch {
  const [searchParams, setSearchParams] = useSearchParams();
  // Clamp ``?limit`` to the backend cap so a hand-edited URL can't 422 the list.
  const limit = readInt(searchParams.get("limit"), defaultLimit, 1, MAX_PAGE_LIMIT);
  const offset = readInt(searchParams.get("offset"), 0, 0);

  const setOffset = (next: number) => {
    const params = new URLSearchParams(searchParams);
    if (next <= 0) params.delete("offset");
    else params.set("offset", String(next));
    setSearchParams(params);
  };

  const setParams = (updates: Record<string, string>) => {
    const params = new URLSearchParams(searchParams);
    for (const [key, value] of Object.entries(updates)) {
      if (value === "") params.delete(key);
      else params.set(key, value);
    }
    params.delete("offset");
    setSearchParams(params);
  };

  return { searchParams, limit, offset, setOffset, setParams };
}
