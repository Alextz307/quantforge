export type SignalKind = "directional" | "leverage";

// Strategies whose generate_signals emit a continuous position-size / leverage
// multiplier rather than a {-1, 0, +1} direction. Mirrors the framework:
// VolatilityTargeting -> [0, max_leverage] (long-only); ReturnForecast ->
// [-max_leverage, +max_leverage] (signed). Every other strategy is directional.
const LEVERAGE_STRATEGIES: ReadonlySet<string> = new Set(["VolatilityTargeting", "ReturnForecast"]);

export function signalKindForStrategy(strategyName: string): SignalKind {
  return LEVERAGE_STRATEGIES.has(strategyName) ? "leverage" : "directional";
}
