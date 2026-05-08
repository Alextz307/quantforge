import { useEffect, useState } from "react";
import type { StrategySchema } from "@/api/strategies";

export type StrategyParamsState = [
  Record<string, unknown>,
  (next: Record<string, unknown>) => void,
];

/**
 * Resets to the schema's canonical defaults when the strategy changes.
 * Gated on ``schemaData?.qualname`` (string) — not the schema object — so
 * background refetches that return the same schema don't nuke user edits.
 */
export function useStrategyParamsState(
  strategyName: string,
  schemaData: StrategySchema | undefined,
): StrategyParamsState {
  const [params, setParams] = useState<Record<string, unknown>>({});
  const schemaQualname = schemaData?.qualname;
  useEffect(() => {
    setParams({ ...(schemaData?.canonical_params ?? {}) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyName, schemaQualname]);
  return [params, setParams];
}
