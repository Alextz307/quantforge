import { describe, expect, it } from "vitest";
import { extractApiError } from "@/api/errors";

const FALLBACK = "Request failed";

describe("extractApiError", () => {
  it("returns fallback for non-object errors", () => {
    expect(extractApiError(undefined, FALLBACK)).toBe(FALLBACK);
    expect(extractApiError(null, FALLBACK)).toBe(FALLBACK);
    expect(extractApiError("oops", FALLBACK)).toBe(FALLBACK);
    expect(extractApiError(42, FALLBACK)).toBe(FALLBACK);
  });

  it("surfaces a string detail (FastAPI HTTPException)", () => {
    expect(extractApiError({ detail: "Invalid credentials" }, FALLBACK)).toBe(
      "Invalid credentials",
    );
  });

  it("formats Pydantic validation errors with field name + message", () => {
    const error = {
      detail: [
        {
          loc: ["body", "password"],
          msg: "String should have at least 8 characters",
          type: "string_too_short",
        },
      ],
    };

    expect(extractApiError(error, FALLBACK)).toBe(
      "password: String should have at least 8 characters",
    );
  });

  it("joins multiple validation errors with semicolons", () => {
    const error = {
      detail: [
        { loc: ["body", "username"], msg: "Field required" },
        { loc: ["body", "password"], msg: "String should have at least 8 characters" },
      ],
    };

    expect(extractApiError(error, FALLBACK)).toBe(
      "username: Field required; password: String should have at least 8 characters",
    );
  });

  it("falls back when validation array is empty or malformed", () => {
    expect(extractApiError({ detail: [] }, FALLBACK)).toBe(FALLBACK);
    expect(extractApiError({ detail: [{ unrelated: true }] }, FALLBACK)).toBe(FALLBACK);
  });
});
