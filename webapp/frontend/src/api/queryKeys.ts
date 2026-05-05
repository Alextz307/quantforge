export const queryKeys = {
  me: ["auth", "me"] as const,
  users: ["users"] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["runs", id] as const,
  runFolds: (id: string) => ["runs", id, "folds"] as const,
} as const;
