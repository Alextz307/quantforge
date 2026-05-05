import { http, HttpResponse } from "msw";
import { API_PATHS, toMswPath } from "@/api/paths";
import type { FoldRow, RunDetail, RunSummary } from "@/api/runs";
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
    return new HttpResponse(null, { status: 404 });
  }),
];
