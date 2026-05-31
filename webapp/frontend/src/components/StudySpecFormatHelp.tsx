import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { useStudySpecSchema } from "@/api/studyUploads";
import { useConfigList } from "@/api/configs";
import { STUDY_LEG_SKELETON } from "@/lib/studySpecSkeleton";

interface FieldRow {
  path: string;
  type: string;
  required: boolean;
  default: string;
  description: string;
}

interface StudySpecFormatHelpProps {
  onInsertLeg: (snippet: string) => void;
}

/**
 * Side-panel reference shown next to the YAML editor.
 *
 * Three sections: a field reference table auto-derived from
 * ``GET /api/configs/study_spec/schema`` (single source of truth - the same
 * descriptions hover-show inside ``mypy`` and surface on the Pydantic
 * models), the canonical 1-leg example, and quick-action buttons.
 *
 * The schema fetch is cached forever (the JSON Schema is generated from
 * frozen Pydantic models and never changes at runtime), so opening the
 * panel after the first paint is free.
 */
export function StudySpecFormatHelp({ onInsertLeg }: StudySpecFormatHelpProps) {
  const schemaQuery = useStudySpecSchema();
  const universesQuery = useConfigList("universe");
  const [showUniverses, setShowUniverses] = useState(false);

  const rows = useMemo<FieldRow[]>(() => {
    if (!schemaQuery.data) return [];
    return flattenStudySpecSchema(schemaQuery.data);
  }, [schemaQuery.data]);

  return (
    <aside className="flex flex-col gap-4 text-sm">
      <section>
        <h3 className="mb-2 text-sm font-semibold">Quick actions</h3>
        <div className="flex flex-col gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              onInsertLeg(STUDY_LEG_SKELETON);
            }}
          >
            Insert leg template
          </Button>
          <div className="flex flex-col gap-1">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setShowUniverses((prev) => !prev);
              }}
              disabled={universesQuery.isError}
            >
              {showUniverses ? "Hide universes" : "Browse universes"}
            </Button>
            {showUniverses && (
              <div className="rounded-md border border-input bg-muted/50 p-2">
                {universesQuery.isLoading ? (
                  <p className="text-xs text-muted-foreground">Loading...</p>
                ) : universesQuery.isError ? (
                  <p className="text-xs text-destructive">Failed to load universes.</p>
                ) : universesQuery.data && universesQuery.data.length > 0 ? (
                  <ul className="max-h-40 space-y-0.5 overflow-auto font-mono text-xs">
                    {universesQuery.data.map((u) => (
                      <li key={u.name}>{u.name}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    No universes registered under <code>config/universes/</code>.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold">Field reference</h3>
        {schemaQuery.isLoading ? (
          <p className="text-xs text-muted-foreground">Loading schema...</p>
        ) : schemaQuery.isError ? (
          <p className="text-xs text-destructive">Failed to load schema.</p>
        ) : (
          <div className="max-h-72 overflow-auto rounded-md border border-input">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-muted text-left">
                <tr>
                  <th className="px-2 py-1 font-medium">Field</th>
                  <th className="px-2 py-1 font-medium">Type</th>
                  <th className="px-2 py-1 font-medium">Req</th>
                  <th className="px-2 py-1 font-medium">Default</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.path} className="border-t border-input align-top">
                    <td className="px-2 py-1 font-mono">{row.path}</td>
                    <td className="px-2 py-1 font-mono">{row.type}</td>
                    <td className="px-2 py-1">{row.required ? "yes" : ""}</td>
                    <td className="px-2 py-1 font-mono">{row.default || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <ul className="space-y-2 border-t border-input bg-muted/30 px-3 py-2">
              {rows.map((row) => (
                <li key={`desc-${row.path}`} className="text-xs">
                  <span className="font-mono font-medium">{row.path}</span>{" "}
                  <span className="text-muted-foreground">{row.description}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold">Canonical example</h3>
        <pre className="max-h-72 overflow-auto rounded-md border border-input bg-muted/50 p-2 font-mono text-xs leading-snug">
          {EXAMPLE_YAML}
        </pre>
      </section>
    </aside>
  );
}

/** Canonical multi-leg example - sized to fit alongside the editor. */
const EXAMPLE_YAML = `name: main_study
description: Empirical sweep across strategies and universes.
seed: 42
output_dir: studies/main

legs:
  - strategy: AdaptiveBollinger
    strategy_config: config/strategies/adaptive_bollinger.yaml
    hpo_config: config/hpo/adaptive_bollinger.yaml
    universes:
      - spy_daily_5y
      - qqq_daily_5y
      - gld_daily_5y

  - strategy: VolatilityTargeting
    strategy_config: config/strategies/volatility_targeting.yaml
    hpo_config: config/hpo/volatility_targeting.yaml
    universes:
      - spy_daily_5y
      - qqq_daily_5y
`;

/**
 * Walk the StudySpec JSON Schema (Pydantic-emitted) and flatten it to one row
 * per top-level + ``StudyLeg[*]`` field. We deliberately stop one level deep:
 * deeper Pydantic types (``ComponentConfig``, ``ValidationConfig``) don't
 * appear in study specs and surfacing them would dilute the reference.
 */
function flattenStudySpecSchema(schema: Record<string, unknown>): FieldRow[] {
  const defs = (schema["$defs"] ?? schema["definitions"]) as
    | Record<string, Record<string, unknown>>
    | undefined;
  const topProps = (schema["properties"] ?? {}) as Record<string, Record<string, unknown>>;
  const required = new Set((schema["required"] as string[] | undefined) ?? []);

  const rows: FieldRow[] = [];
  for (const [name, prop] of Object.entries(topProps)) {
    rows.push({
      path: name,
      type: renderType(prop),
      required: required.has(name),
      default: renderDefault(prop),
      description: (prop["description"] as string | undefined) ?? "",
    });
  }
  const legDef = defs?.["StudyLeg"];
  if (legDef) {
    const legProps = (legDef["properties"] ?? {}) as Record<string, Record<string, unknown>>;
    const legRequired = new Set((legDef["required"] as string[] | undefined) ?? []);
    for (const [name, prop] of Object.entries(legProps)) {
      rows.push({
        path: `legs[*].${name}`,
        type: renderType(prop),
        required: legRequired.has(name),
        default: renderDefault(prop),
        description: (prop["description"] as string | undefined) ?? "",
      });
    }
  }
  return rows;
}

function renderType(prop: Record<string, unknown>): string {
  if (typeof prop["type"] === "string") {
    if (prop["type"] === "array") {
      const items = prop["items"] as Record<string, unknown> | undefined;
      const inner = items ? renderType(items) : "any";
      return `${inner}[]`;
    }
    return prop["type"];
  }
  const refValue = prop["$ref"];
  if (typeof refValue === "string") {
    return refValue.split("/").pop() ?? "object";
  }
  if (Array.isArray(prop["anyOf"])) {
    const types = (prop["anyOf"] as Record<string, unknown>[])
      .map(renderType)
      .filter((t) => t !== "null");
    return types.join(" | ") || "any";
  }
  return "any";
}

function renderDefault(prop: Record<string, unknown>): string {
  if (!("default" in prop)) return "";
  const value = prop["default"];
  if (value === null) return "null";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}
