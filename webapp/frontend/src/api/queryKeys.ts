import type { components } from "./client";

export interface RunsPageParams {
  limit: number;
  offset: number;
  sortBy: components["schemas"]["RunSortBy"];
  order: components["schemas"]["SortOrder"];
  strategy?: string;
  ticker?: string;
  since?: string;
}

export const queryKeys = {
  me: ["auth", "me"] as const,
  users: ["users"] as const,
  runs: ["runs"] as const,
  runsPage: (params: RunsPageParams & { allUsers: boolean }) => ["runs", "page", params] as const,
  run: (id: string) => ["runs", id] as const,
  runFolds: (id: string) => ["runs", id, "folds"] as const,
  comparisons: ["comparisons"] as const,
  comparisonsList: (allUsers: boolean) => ["comparisons", { allUsers }] as const,
  comparison: (name: string) => ["comparisons", name] as const,
  holdoutEvals: ["holdoutEvals"] as const,
  holdoutEvalsList: (allUsers: boolean) => ["holdoutEvals", { allUsers }] as const,
  holdoutEval: (name: string) => ["holdoutEvals", name] as const,
  studies: ["studies"] as const,
  studiesList: (allUsers: boolean) => ["studies", { allUsers }] as const,
  study: (name: string) => ["studies", name] as const,
  studyConsolidated: (name: string) => ["studies", name, "consolidated"] as const,
  hpoStudies: ["hpoStudies"] as const,
  hpoStudiesList: (allUsers: boolean) => ["hpoStudies", { allUsers }] as const,
  hpoStudy: (name: string) => ["hpoStudies", name] as const,
  hpoTrials: (name: string) => ["hpoStudies", name, "trials"] as const,
  hpoParamImportance: (name: string) => ["hpoStudies", name, "param-importance"] as const,
  jobsAll: ["jobs"] as const,
  jobs: (params: { allUsers: boolean }) => ["jobs", { allUsers: params.allUsers }] as const,
  job: (id: string) => ["jobs", id] as const,
  configs: (kind: string) => ["configs", kind] as const,
  configDetail: (kind: string, name: string) => ["configs", kind, name] as const,
  studySpecSchema: ["configs", "studySpec", "schema"] as const,
  studyUploads: ["configs", "studyUploads"] as const,
  studyUpload: (slug: string) => ["configs", "studyUploads", slug] as const,
  universeSpecSchema: ["configs", "universeSpec", "schema"] as const,
  universeUploads: ["configs", "universeUploads"] as const,
  universeUpload: (slug: string) => ["configs", "universeUploads", slug] as const,
  strategies: ["strategies"] as const,
  strategySchema: (name: string) => ["strategies", name, "schema"] as const,
  deployments: ["deployments"] as const,
  deploymentsList: (allUsers: boolean) => ["deployments", { allUsers }] as const,
  deployment: (id: string) => ["deployments", id] as const,
  deploymentSignals: (id: string) => ["deployments", id, "signals"] as const,
} as const;
