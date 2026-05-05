import type { components } from "@/api/client";

export type SourceKind = components["schemas"]["HoldoutEvalSummary"]["source_kind"];

export const SOURCE_KIND_RUN: SourceKind = "run";
export const SOURCE_KIND_HPO: SourceKind = "hpo";

export const SOURCE_KINDS = [
  SOURCE_KIND_RUN,
  SOURCE_KIND_HPO,
] as const satisfies readonly SourceKind[];

export function isRunSource(kind: SourceKind): boolean {
  return kind === SOURCE_KIND_RUN;
}

export function isHpoSource(kind: SourceKind): boolean {
  return kind === SOURCE_KIND_HPO;
}

export function sourceKindLabel(kind: SourceKind): string {
  return kind === SOURCE_KIND_RUN ? "Run" : "HPO trial";
}
