// Drift guard: zod schemas under src/lib/schemas/ must agree with the
// Pydantic shape captured in schema-mirror.snapshot.json (regenerated via
// scripts/check_webapp_schema_mirror.py).

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { z } from "zod";
import { loginSchema } from "@/lib/schemas/login";
import { userCreateSchema } from "@/lib/schemas/userCreate";

interface FieldShape {
  type: string;
  min?: number;
  max?: number;
  default?: unknown;
}
type ModelShape = Record<string, FieldShape>;
type Snapshot = Record<string, ModelShape>;

const SNAPSHOT_PATH = path.resolve(__dirname, "../../../schema-mirror.snapshot.json");
const snapshot = JSON.parse(fs.readFileSync(SNAPSHOT_PATH, "utf-8")) as Snapshot;

function shapeOf(schema: z.ZodObject<z.ZodRawShape>): ModelShape {
  const out: ModelShape = {};
  for (const [name, field] of Object.entries(schema.shape)) {
    out[name] = describeField(field);
  }
  return out;
}

function describeField(field: z.ZodTypeAny): FieldShape {
  // Unwrap defaults / optional wrappers down to the inner type.
  let inner = field;
  let defaultValue: unknown;
  let unwrapped = true;
  while (unwrapped) {
    unwrapped = false;
    const def = inner._def as {
      typeName?: string;
      defaultValue?: () => unknown;
      innerType?: z.ZodTypeAny;
    };
    if (def.typeName === "ZodDefault" && def.defaultValue && def.innerType) {
      defaultValue = def.defaultValue();
      inner = def.innerType;
      unwrapped = true;
    } else if (def.typeName === "ZodOptional" && def.innerType) {
      inner = def.innerType;
      unwrapped = true;
    }
  }

  const def = inner._def as {
    typeName?: string;
    checks?: Array<{ kind: string; value?: number }>;
    values?: readonly string[];
  };
  const result: FieldShape = { type: zodTypeName(inner) };
  if (defaultValue !== undefined) {
    result.default = defaultValue;
  }
  if (def.checks) {
    for (const check of def.checks) {
      if (check.kind === "min" && typeof check.value === "number") result.min = check.value;
      if (check.kind === "max" && typeof check.value === "number") result.max = check.value;
    }
  }
  return result;
}

function zodTypeName(field: z.ZodTypeAny): string {
  const def = field._def as { typeName?: string; values?: readonly string[] };
  if (def.typeName === "ZodString") return "string";
  if (def.typeName === "ZodNumber") return "number";
  if (def.typeName === "ZodBoolean") return "boolean";
  if (def.typeName === "ZodEnum" && def.values) {
    return "enum:" + [...def.values].sort().join("|");
  }
  return def.typeName ?? "unknown";
}

describe("Pydantic <-> zod schema mirror", () => {
  it("login zod schema matches the snapshot", () => {
    expect(shapeOf(loginSchema)).toEqual(snapshot.login);
  });

  it("userCreate zod schema matches the snapshot", () => {
    expect(shapeOf(userCreateSchema)).toEqual(snapshot.userCreate);
  });

  it("snapshot lists exactly the mirrored models", () => {
    expect(Object.keys(snapshot).sort()).toEqual(["login", "userCreate"]);
  });
});
