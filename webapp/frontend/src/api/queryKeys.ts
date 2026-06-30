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

export interface HoldoutEvalsPageParams {
  limit: number;
  offset: number;
  sortBy: components["schemas"]["HoldoutSortBy"];
  order: components["schemas"]["SortOrder"];
  sourceKind?: components["schemas"]["HoldoutEvalSummary"]["source_kind"];
  since?: string;
}

export interface HpoStudiesPageParams {
  limit: number;
  offset: number;
  sortBy: components["schemas"]["HpoSortBy"];
  order: components["schemas"]["SortOrder"];
  store?: string;
  since?: string;
}

export interface ComparisonsPageParams {
  limit: number;
  offset: number;
  strategy?: string;
  since?: string;
}

export interface StudiesPageParams {
  limit: number;
  offset: number;
  spec?: string;
  since?: string;
}

export const queryKeys = {
  me: ["auth", "me"] as const,
  users: ["users"] as const,
  runs: ["runs"] as const,
  runsPage: (params: RunsPageParams & { allUsers: boolean }) => ["runs", "page", params] as const,
  run: (id: string) => ["runs", id] as const,
  runFolds: (id: string) => ["runs", id, "folds"] as const,
  runFeatureImportance: (id: string) => ["runs", id, "feature-importance"] as const,
  comparisons: ["comparisons"] as const,
  comparisonsPage: (params: ComparisonsPageParams & { allUsers: boolean }) =>
    ["comparisons", "page", params] as const,
  comparison: (name: string) => ["comparisons", name] as const,
  holdoutEvals: ["holdoutEvals"] as const,
  holdoutEvalsPage: (params: HoldoutEvalsPageParams & { allUsers: boolean }) =>
    ["holdoutEvals", "page", params] as const,
  holdoutEvalsPicker: (params: { allUsers: boolean }) =>
    ["holdoutEvals", "picker", params] as const,
  holdoutEval: (name: string) => ["holdoutEvals", name] as const,
  studies: ["studies"] as const,
  studiesPage: (params: StudiesPageParams & { allUsers: boolean }) =>
    ["studies", "page", params] as const,
  study: (name: string) => ["studies", name] as const,
  studyConsolidated: (name: string) => ["studies", name, "consolidated"] as const,
  hpoStudies: ["hpoStudies"] as const,
  hpoStudiesPage: (params: HpoStudiesPageParams & { allUsers: boolean }) =>
    ["hpoStudies", "page", params] as const,
  hpoStudiesPicker: (params: { allUsers: boolean }) => ["hpoStudies", "picker", params] as const,
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
  deploymentEvaluation: (id: string, cost: string) =>
    ["deployments", id, "evaluation", cost] as const,
} as const;
