// Drift guard: the backend fixes every list endpoint's ``limit`` default and
// cap in one place (``DEFAULT_PAGE_LIMIT`` / ``MAX_PAGE_LIMIT`` in
// schemas/pagination.py, wired as FastAPI ``Query(default, le=...)``). The
// frontend mirrors both - ``MAX_PAGE_LIMIT`` in api/client.ts (URL-limit clamp)
// and ``DEFAULT_PAGE_LIMIT`` in hooks/usePaginatedSearch.ts (initial page size).
// This asserts the two pairs agree against the committed OpenAPI snapshot
// (regenerated from the backend), so changing a backend constant without
// updating the frontend fails here instead of silently 422-ing hand-edited
// URLs or paging in the wrong stride.

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { MAX_PAGE_LIMIT } from "@/api/client";
import { DEFAULT_PAGE_LIMIT } from "@/hooks/usePaginatedSearch";

const SNAPSHOT_PATH = path.resolve(__dirname, "../../openapi.snapshot.json");

interface OpenApiParam {
  name: string;
  schema?: { maximum?: number; default?: number };
}
type Snapshot = {
  paths: Record<string, Record<string, { parameters?: OpenApiParam[] }>>;
};

// The ``limit`` param schemas across every list endpoint. Only the paginated
// list endpoints carry a numeric maximum + default; an unrelated ``limit`` (the
// deployment-signals endpoint) has neither and is excluded by the callers'
// numeric filters.
function listLimitSchemas(): { maximum?: number; default?: number }[] {
  const snapshot = JSON.parse(fs.readFileSync(SNAPSHOT_PATH, "utf-8")) as Snapshot;
  const schemas: { maximum?: number; default?: number }[] = [];
  for (const operations of Object.values(snapshot.paths)) {
    for (const operation of Object.values(operations)) {
      for (const param of operation.parameters ?? []) {
        if (param.name === "limit" && param.schema !== undefined) {
          schemas.push(param.schema);
        }
      }
    }
  }
  return schemas;
}

describe("page-limit constants", () => {
  it("MAX_PAGE_LIMIT matches every list endpoint's limit cap in the snapshot", () => {
    const maxima = listLimitSchemas()
      .map((s) => s.maximum)
      .filter((m): m is number => typeof m === "number");
    expect(maxima.length).toBeGreaterThan(0);
    for (const max of maxima) {
      expect(max).toBe(MAX_PAGE_LIMIT);
    }
  });

  it("DEFAULT_PAGE_LIMIT matches every list endpoint's limit default in the snapshot", () => {
    const defaults = listLimitSchemas()
      .map((s) => s.default)
      .filter((d): d is number => typeof d === "number");
    expect(defaults.length).toBeGreaterThan(0);
    for (const dflt of defaults) {
      expect(dflt).toBe(DEFAULT_PAGE_LIMIT);
    }
  });
});
