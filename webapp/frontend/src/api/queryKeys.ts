export const queryKeys = {
  me: ["auth", "me"] as const,
  users: ["users"] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["runs", id] as const,
  runFolds: (id: string) => ["runs", id, "folds"] as const,
  comparisons: ["comparisons"] as const,
  comparison: (name: string) => ["comparisons", name] as const,
  holdoutEvals: ["holdoutEvals"] as const,
  holdoutEval: (name: string) => ["holdoutEvals", name] as const,
} as const;
