import { http, HttpResponse } from "msw";
import type { ComparisonDetail, ComparisonSummary } from "@/api/comparisons";
import type { HoldoutEvalDetail, HoldoutEvalSummary } from "@/api/holdout";
import type { HpoDetail, HpoSummary, TrialRow } from "@/api/hpo";
import { API_PATHS, toMswPath } from "@/api/paths";
import type {
  RegimeReportDetail,
  RegimeReportSummary,
} from "@/api/regime";
import type { FoldRow, RunDetail, RunSummary } from "@/api/runs";
import type {
  StudyConsolidatedDTO,
  StudyDetail,
  StudySummary,
} from "@/api/studies";
import { ROLE_ADMIN, ROLE_USER, type UserCreate, type UserPublic } from "@/api/users";

export const ADMIN_USER: UserPublic = { id: 1, username: "alex", role: ROLE_ADMIN };
export const REGULAR_USER: UserPublic = { id: 2, username: "guest", role: ROLE_USER };

export const SEED_USERS: UserPublic[] = [ADMIN_USER, REGULAR_USER];

export const RUN_SPY: RunSummary = {
  experiment_id: "exp_spy",
  name: "spy_daily_5y_ab",
  strategy: "AdaptiveBollinger",
  tickers: ["SPY"],
  interval: "DAY",
  store: "thesis_demo",
  created_at: "2026-04-01T12:00:00Z",
  sharpe_mean: 1.234,
  calmar_mean: 0.456,
};

export const RUN_IVV_VOO: RunSummary = {
  experiment_id: "exp_pairs",
  name: "ivv_voo_pairs",
  strategy: "PairsTrading",
  tickers: ["IVV", "VOO"],
  interval: "DAY",
  store: "thesis_demo",
  created_at: "2026-04-15T12:00:00Z",
  sharpe_mean: 2.0,
  calmar_mean: 1.1,
};

export const SEED_RUNS: RunSummary[] = [RUN_SPY, RUN_IVV_VOO];

export const RUN_SPY_DETAIL: RunDetail = {
  experiment_id: RUN_SPY.experiment_id,
  name: RUN_SPY.name,
  strategy: RUN_SPY.strategy,
  tickers: RUN_SPY.tickers,
  interval: RUN_SPY.interval,
  store: RUN_SPY.store,
  created_at: RUN_SPY.created_at,
  git_sha: "0123456789abcdef0123456789abcdef01234567",
  seed: 42,
  data_hash: "deadbeefcafebabe1234567890abcdef",
  slippage_scenario: "normal",
  holdout_start: "2026-01-01T00:00:00Z",
  pretrained_leaves: [],
  metrics: { sharpe_mean: 1.234, calmar_mean: 0.456, max_drawdown_mean: -0.12 },
  plots: ["equity.png", "fold_stability.svg"],
};

export const RUN_SPY_FOLDS: FoldRow[] = [
  {
    fold_index: 0,
    train_start: "2021-01-01T00:00:00Z",
    train_end: "2023-01-01T00:00:00Z",
    test_start: "2023-01-02T00:00:00Z",
    test_end: "2024-01-01T00:00:00Z",
    total_return: 0.15,
    annualized_return: 0.14,
    annualized_volatility: 0.18,
    sharpe_ratio: 1.1,
    sortino_ratio: 1.4,
    calmar_ratio: 0.9,
    max_drawdown: -0.08,
    win_rate: 0.55,
    trade_count: 42,
    equity_curve: [1.0, 1.05, 1.1, 1.15],
  },
  {
    fold_index: 1,
    train_start: "2022-01-01T00:00:00Z",
    train_end: "2024-01-01T00:00:00Z",
    test_start: "2024-01-02T00:00:00Z",
    test_end: "2025-01-01T00:00:00Z",
    total_return: 0.08,
    annualized_return: 0.08,
    annualized_volatility: 0.15,
    sharpe_ratio: 0.6,
    sortino_ratio: 0.8,
    calmar_ratio: 0.5,
    max_drawdown: -0.06,
    win_rate: 0.51,
    trade_count: 39,
    equity_curve: [1.0, 1.02, 1.05, 1.08],
  },
];

export const RUN_PAIRS_FOLDS: FoldRow[] = [
  {
    fold_index: 0,
    train_start: "2021-01-01T00:00:00Z",
    train_end: "2023-01-01T00:00:00Z",
    test_start: "2023-01-02T00:00:00Z",
    test_end: "2024-01-01T00:00:00Z",
    total_return: 0.22,
    annualized_return: 0.21,
    annualized_volatility: 0.12,
    sharpe_ratio: 1.7,
    sortino_ratio: 2.0,
    calmar_ratio: 1.6,
    max_drawdown: -0.04,
    win_rate: 0.6,
    trade_count: 30,
    equity_curve: [1.0, 1.07, 1.15, 1.22],
  },
];

export const COMPARISON_DEMO_SUMMARY: ComparisonSummary = {
  name: "demo_comparison_2026Q1",
  store: "thesis_demo",
  created_at: "2026-04-20T12:00:00Z",
  strategies: [RUN_SPY.strategy, RUN_IVV_VOO.strategy],
};

export const COMPARISON_DEMO_DETAIL: ComparisonDetail = {
  name: COMPARISON_DEMO_SUMMARY.name,
  store: COMPARISON_DEMO_SUMMARY.store,
  created_at: COMPARISON_DEMO_SUMMARY.created_at,
  git_sha: "fedcba9876543210fedcba9876543210fedcba98",
  per_strategy_stats: [
    {
      strategy: RUN_SPY.strategy,
      experiment_id: RUN_SPY.experiment_id,
      n_folds: 2,
      sharpe_mean: 0.85,
      sharpe_std: 0.25,
      sharpe_ci95_low: 0.4,
      sharpe_ci95_high: 1.3,
      sortino_mean: 1.1,
      sortino_std: 0.3,
      sortino_ci95_low: 0.6,
      sortino_ci95_high: 1.6,
      calmar_mean: 0.7,
      calmar_std: 0.2,
      calmar_ci95_low: 0.4,
      calmar_ci95_high: 1.0,
      total_return_mean: 0.115,
      total_return_std: 0.035,
      max_drawdown_mean: -0.07,
      max_drawdown_worst: -0.08,
      win_rate_mean: 0.53,
      trade_count_total: 81,
    },
    {
      strategy: RUN_IVV_VOO.strategy,
      experiment_id: RUN_IVV_VOO.experiment_id,
      n_folds: 1,
      sharpe_mean: 1.7,
      sharpe_std: 0.0,
      sharpe_ci95_low: 1.7,
      sharpe_ci95_high: 1.7,
      sortino_mean: 2.0,
      sortino_std: 0.0,
      sortino_ci95_low: 2.0,
      sortino_ci95_high: 2.0,
      calmar_mean: 1.6,
      calmar_std: 0.0,
      calmar_ci95_low: 1.6,
      calmar_ci95_high: 1.6,
      total_return_mean: 0.22,
      total_return_std: 0.0,
      max_drawdown_mean: -0.04,
      max_drawdown_worst: -0.04,
      win_rate_mean: 0.6,
      trade_count_total: 30,
    },
  ],
  plots: ["ranking.png", "equity_overlay.svg"],
};

export const SEED_COMPARISONS: ComparisonSummary[] = [COMPARISON_DEMO_SUMMARY];

export const HOLDOUT_DEMO_SUMMARY: HoldoutEvalSummary = {
  name: "demo_holdout_spy",
  store: "thesis_demo",
  created_at: "2026-04-25T12:00:00Z",
  source_kind: "run",
  source_id: RUN_SPY.experiment_id,
  holdout_start: "2026-01-01T00:00:00Z",
};

export const HOLDOUT_DEMO_DETAIL: HoldoutEvalDetail = {
  name: HOLDOUT_DEMO_SUMMARY.name,
  store: HOLDOUT_DEMO_SUMMARY.store,
  created_at: HOLDOUT_DEMO_SUMMARY.created_at,
  git_sha: "abcdef0123456789abcdef0123456789abcdef01",
  source_kind: HOLDOUT_DEMO_SUMMARY.source_kind,
  source_id: HOLDOUT_DEMO_SUMMARY.source_id,
  source_path: "experiment_results/runs/exp_spy",
  holdout_start: HOLDOUT_DEMO_SUMMARY.holdout_start,
  data_hash: "0123456789abcdef0123456789abcdef",
  n_dev_bars: 1200,
  n_holdout_bars: 250,
  slippage_scenario: "normal",
  total_return: 0.07,
  annualized_return: 0.09,
  annualized_volatility: 0.16,
  sharpe_ratio: 0.55,
  sortino_ratio: 0.7,
  calmar_ratio: 0.4,
  max_drawdown: -0.05,
  win_rate: 0.52,
  trade_count: 18,
  equity_curve: [1.0, 1.01, 1.03, 1.05, 1.07],
  plots: ["holdout_equity.png"],
};

export const SEED_HOLDOUT_EVALS: HoldoutEvalSummary[] = [HOLDOUT_DEMO_SUMMARY];

export const REGIME_DEMO_SUMMARY: RegimeReportSummary = {
  name: "demo_regime_volatility",
  store: "thesis_demo",
  created_at: "2026-04-22T12:00:00Z",
  experiment_id: RUN_SPY.experiment_id,
  kind: "volatility",
  detector_name: "VolatilityRegime",
  regime_labels: ["LOW", "HIGH"],
};

export const REGIME_DEMO_DETAIL: RegimeReportDetail = {
  name: REGIME_DEMO_SUMMARY.name,
  store: REGIME_DEMO_SUMMARY.store,
  created_at: REGIME_DEMO_SUMMARY.created_at,
  git_sha: "11111111111111111111111111111111111111aa",
  experiment_id: REGIME_DEMO_SUMMARY.experiment_id,
  kind: REGIME_DEMO_SUMMARY.kind,
  detector_name: REGIME_DEMO_SUMMARY.detector_name,
  per_regime_stats: [
    {
      regime_label: "LOW",
      n_folds: 4,
      sharpe_mean: 1.2,
      sharpe_std: 0.3,
      sharpe_ci95_low: 0.6,
      sharpe_ci95_high: 1.8,
      sortino_mean: 1.4,
      sortino_std: 0.3,
      sortino_ci95_low: 0.8,
      sortino_ci95_high: 2.0,
      calmar_mean: 1.0,
      calmar_std: 0.2,
      calmar_ci95_low: 0.7,
      calmar_ci95_high: 1.4,
      total_return_mean: 0.18,
      total_return_std: 0.04,
      max_drawdown_mean: -0.05,
      max_drawdown_worst: -0.08,
      win_rate_mean: 0.58,
      trade_count_total: 100,
    },
    {
      regime_label: "HIGH",
      n_folds: 3,
      sharpe_mean: 0.4,
      sharpe_std: 0.5,
      sharpe_ci95_low: -0.4,
      sharpe_ci95_high: 1.2,
      sortino_mean: 0.5,
      sortino_std: 0.5,
      sortino_ci95_low: -0.3,
      sortino_ci95_high: 1.3,
      calmar_mean: 0.2,
      calmar_std: 0.3,
      calmar_ci95_low: -0.2,
      calmar_ci95_high: 0.7,
      total_return_mean: 0.05,
      total_return_std: 0.06,
      max_drawdown_mean: -0.12,
      max_drawdown_worst: -0.18,
      win_rate_mean: 0.46,
      trade_count_total: 60,
    },
  ],
  per_regime_fold_indices: { LOW: [0, 2, 4, 6], HIGH: [1, 3, 5] },
  mixed_fold_indices: [],
  slices: [
    { label: "LOW", start: "2024-01-01T00:00:00Z", end: "2024-06-01T00:00:00Z" },
    { label: "HIGH", start: "2024-06-01T00:00:00Z", end: "2024-09-01T00:00:00Z" },
    { label: "LOW", start: "2024-09-01T00:00:00Z", end: "2025-01-01T00:00:00Z" },
  ],
  plots: ["timeline.png", "metric_heatmap.svg"],
};

export const SEED_REGIME_REPORTS: RegimeReportSummary[] = [REGIME_DEMO_SUMMARY];

export const STUDY_DEMO_SUMMARY: StudySummary = {
  name: "main",
  spec_name: "demo_spec",
  started_at: "2026-04-01T00:00:00Z",
  total_legs: 4,
  completed_legs: 2,
  completion_pct: 50,
};

export const STUDY_DEMO_DETAIL: StudyDetail = {
  name: STUDY_DEMO_SUMMARY.name,
  spec_name: STUDY_DEMO_SUMMARY.spec_name,
  spec_hash: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
  started_at: STUDY_DEMO_SUMMARY.started_at,
  total_legs: STUDY_DEMO_SUMMARY.total_legs,
  completed_legs: STUDY_DEMO_SUMMARY.completed_legs,
  completion_pct: STUDY_DEMO_SUMMARY.completion_pct,
  cross_strategy_compares_done: ["spy_daily_5y"],
  legs: [
    {
      leg_id: "AdaptiveBollinger__spy_daily_5y",
      strategy: "AdaptiveBollinger",
      universe: "spy_daily_5y",
      is_complete: true,
      error: null,
      run_experiment_id: RUN_SPY.experiment_id,
      started_at: "2026-04-01T00:00:00Z",
      completed_at: "2026-04-01T01:00:00Z",
      steps_completed: ["tune", "run", "regime", "holdout_eval"],
    },
    {
      leg_id: "AdaptiveBollinger__spy_daily_10y",
      strategy: "AdaptiveBollinger",
      universe: "spy_daily_10y",
      is_complete: false,
      error: null,
      run_experiment_id: null,
      started_at: null,
      completed_at: null,
      steps_completed: [],
    },
    {
      leg_id: "PairsTrading__spy_daily_5y",
      strategy: "PairsTrading",
      universe: "spy_daily_5y",
      is_complete: true,
      error: null,
      run_experiment_id: RUN_IVV_VOO.experiment_id,
      started_at: "2026-04-01T02:00:00Z",
      completed_at: "2026-04-01T03:00:00Z",
      steps_completed: ["tune", "run", "regime", "holdout_eval"],
    },
    {
      leg_id: "PairsTrading__spy_daily_10y",
      strategy: "PairsTrading",
      universe: "spy_daily_10y",
      is_complete: false,
      error: "data fetch failed",
      run_experiment_id: null,
      started_at: "2026-04-01T04:00:00Z",
      completed_at: null,
      steps_completed: ["tune"],
    },
  ],
};

export const STUDY_CONSOLIDATED_DEMO: StudyConsolidatedDTO = {
  study_name: STUDY_DEMO_SUMMARY.name,
  publish_label: "demo_publish",
  created_at: "2026-04-05T00:00:00Z",
  git_sha: "22222222222222222222222222222222222222bb",
  strategies: ["AdaptiveBollinger", "PairsTrading"],
  universes: ["spy_daily_5y", "spy_daily_10y"],
  incomplete_leg_ids: ["AdaptiveBollinger__spy_daily_10y", "PairsTrading__spy_daily_10y"],
  n_legs_with_regime: 2,
  n_legs_with_holdout: 1,
  n_universes_with_pairwise: 1,
  tables: ["master_ranking.tex", "per_regime_ranking.tex"],
  plots: ["strategy_x_universe_heatmap.png", "holdout_dev_scatter.png"],
};

export const SEED_STUDIES: StudySummary[] = [STUDY_DEMO_SUMMARY];

export const HPO_DEMO_SUMMARY: HpoSummary = {
  name: "AdaptiveBollinger__spy_daily_5y",
  store: "studies/main",
  created_at: "2026-04-03T00:00:00Z",
  n_trials: 3,
  n_complete: 2,
  best_value: 0.85,
  best_trial_number: 1,
  direction: "maximize",
};

export const HPO_DEMO_DETAIL: HpoDetail = {
  ...HPO_DEMO_SUMMARY,
  best_config: {
    name: "demo",
    seed: 42,
    strategy: { name: "AdaptiveBollinger", params: { window: 30, k: 2.0 } },
  },
};

export const HPO_DEMO_TRIALS: TrialRow[] = [
  {
    number: 0,
    state: "COMPLETE",
    value: 0.6,
    params: { window: 30, k: 1.5 },
    datetime_start: "2026-04-03T00:00:00Z",
    datetime_complete: "2026-04-03T00:05:00Z",
    experiment_id: "20260403_000000_synthetic_abc_0000",
  },
  {
    number: 1,
    state: "COMPLETE",
    value: 0.85,
    params: { window: 35, k: 2.0 },
    datetime_start: "2026-04-03T00:05:00Z",
    datetime_complete: "2026-04-03T00:10:00Z",
    experiment_id: "20260403_000000_synthetic_abc_0001",
  },
  {
    number: 2,
    state: "FAIL",
    value: null,
    params: { window: 40, k: 2.5 },
    datetime_start: "2026-04-03T00:10:00Z",
    datetime_complete: null,
    experiment_id: null,
  },
];

export const SEED_HPO_STUDIES: HpoSummary[] = [HPO_DEMO_SUMMARY];

export const handlers = [
  http.get("/api/auth/me", () => HttpResponse.json(ADMIN_USER)),
  http.post("/api/auth/login", () => HttpResponse.json(ADMIN_USER)),
  http.post("/api/auth/logout", () => new HttpResponse(null, { status: 204 })),
  http.get("/api/users", () => HttpResponse.json(SEED_USERS)),
  http.post("/api/users", async ({ request }) => {
    const body = (await request.json()) as UserCreate;
    return HttpResponse.json({ id: 99, username: body.username, role: body.role });
  }),
  http.delete("/api/users/:id", () => new HttpResponse(null, { status: 204 })),
  http.get(API_PATHS.runs, () => HttpResponse.json(SEED_RUNS)),
  http.get(toMswPath(API_PATHS.run), ({ params }) => {
    if (params.experiment_id === RUN_SPY.experiment_id) return HttpResponse.json(RUN_SPY_DETAIL);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(toMswPath(API_PATHS.runFolds), ({ params }) => {
    if (params.experiment_id === RUN_SPY.experiment_id) return HttpResponse.json(RUN_SPY_FOLDS);
    if (params.experiment_id === RUN_IVV_VOO.experiment_id)
      return HttpResponse.json(RUN_PAIRS_FOLDS);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(API_PATHS.comparisons, () => HttpResponse.json(SEED_COMPARISONS)),
  http.get(toMswPath(API_PATHS.comparison), ({ params }) => {
    if (params.name === COMPARISON_DEMO_SUMMARY.name)
      return HttpResponse.json(COMPARISON_DEMO_DETAIL);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(API_PATHS.holdoutEvals, () => HttpResponse.json(SEED_HOLDOUT_EVALS)),
  http.get(toMswPath(API_PATHS.holdoutEval), ({ params }) => {
    if (params.name === HOLDOUT_DEMO_SUMMARY.name) return HttpResponse.json(HOLDOUT_DEMO_DETAIL);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(API_PATHS.regimeReports, () => HttpResponse.json(SEED_REGIME_REPORTS)),
  http.get(toMswPath(API_PATHS.regimeReport), ({ params }) => {
    if (params.name === REGIME_DEMO_SUMMARY.name) return HttpResponse.json(REGIME_DEMO_DETAIL);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(API_PATHS.studies, () => HttpResponse.json(SEED_STUDIES)),
  http.get(toMswPath(API_PATHS.study), ({ params }) => {
    if (params.name === STUDY_DEMO_SUMMARY.name) return HttpResponse.json(STUDY_DEMO_DETAIL);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(toMswPath(API_PATHS.studyConsolidated), ({ params }) => {
    if (params.name === STUDY_DEMO_SUMMARY.name) return HttpResponse.json(STUDY_CONSOLIDATED_DEMO);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(API_PATHS.hpoStudies, () => HttpResponse.json(SEED_HPO_STUDIES)),
  http.get(toMswPath(API_PATHS.hpoStudy), ({ params }) => {
    if (params.name === HPO_DEMO_SUMMARY.name) return HttpResponse.json(HPO_DEMO_DETAIL);
    return new HttpResponse(null, { status: 404 });
  }),
  http.get(toMswPath(API_PATHS.hpoTrials), ({ params }) => {
    if (params.name === HPO_DEMO_SUMMARY.name) return HttpResponse.json(HPO_DEMO_TRIALS);
    return new HttpResponse(null, { status: 404 });
  }),
];
