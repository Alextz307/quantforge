import { z } from "zod";

// Interval values are mirrored from src.core.types.Interval. Server-side
// validation via POST /api/configs/validate is the source of truth; this
// list exists for the typed dropdown on the Configure page only.
const INTERVAL_VALUES = [
  "second",
  "minute",
  "five_minute",
  "fifteen_minute",
  "hour",
  "daily",
  "weekly",
] as const;

export type IntervalValue = (typeof INTERVAL_VALUES)[number];

export const INTERVAL_OPTIONS: readonly IntervalValue[] = INTERVAL_VALUES;

export function isIntervalValue(value: string): value is IntervalValue {
  return (INTERVAL_VALUES as readonly string[]).includes(value);
}

const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}$/;

export const TICKERS_INPUT_HINT = "Comma- or space-separated, e.g. SPY, QQQ";

// Mirrored from the ExperimentConfig.name backend constraint - imported by
// ExperimentFieldsSection so the input's HTML maxLength matches the server.
export const NAME_MAX = 64;
const DEFAULT_SEED = 42;
const SEED_MIN = 0;
const DEFAULT_N_SPLITS = 3;
const DEFAULT_TEST_SIZE = 252;
const DEFAULT_GAP = 5;
const N_SPLITS_MIN = 2;
const N_SPLITS_MAX = 50;
const TEST_SIZE_MIN = 10;
const TEST_SIZE_MAX = 5000;
const GAP_MIN = 0;
const GAP_MAX = 500;

export const parseStringList = (input: string): string[] =>
  input
    .split(/[\s,]+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);

// Pure object shape - no cross-field refinement, so consumers can ``.extend()``
// it before applying their own ``.refine()`` (Zod's ``ZodEffects`` doesn't
// support ``.extend``). The Tune form reuses this exact base.
export const experimentBaseSchema = z.object({
  name: z
    .string()
    .min(1, "Name is required")
    .max(NAME_MAX, `Name must be at most ${String(NAME_MAX)} characters`),
  seed: z.coerce.number().int().min(SEED_MIN).default(DEFAULT_SEED),
  tickers: z
    .string()
    .min(1, "At least one ticker is required")
    .refine((v) => parseStringList(v).length > 0, "At least one ticker is required"),
  start: z.string().regex(ISO_DATE_REGEX, "Start must be YYYY-MM-DD"),
  end: z.string().regex(ISO_DATE_REGEX, "End must be YYYY-MM-DD"),
  interval: z.enum(INTERVAL_VALUES).default("daily"),
  strategyName: z.string().min(1, "Strategy is required"),
  nSplits: z.coerce.number().int().min(N_SPLITS_MIN).max(N_SPLITS_MAX).default(DEFAULT_N_SPLITS),
  testSize: z.coerce
    .number()
    .int()
    .min(TEST_SIZE_MIN)
    .max(TEST_SIZE_MAX)
    .default(DEFAULT_TEST_SIZE),
  gap: z.coerce.number().int().min(GAP_MIN).max(GAP_MAX).default(DEFAULT_GAP),
  expanding: z.boolean().default(true),
});

export const startBeforeEndRefinement = {
  predicate: (v: { start: string; end: string }) => new Date(v.start) < new Date(v.end),
  message: "Start must be strictly before end",
  path: ["end"] as const,
};

export const configureFormSchema = experimentBaseSchema.refine(startBeforeEndRefinement.predicate, {
  message: startBeforeEndRefinement.message,
  path: [...startBeforeEndRefinement.path],
});

export type ConfigureFormValues = z.infer<typeof configureFormSchema>;

export const EXPERIMENT_FORM_DEFAULTS: ConfigureFormValues = {
  name: "",
  seed: DEFAULT_SEED,
  tickers: "",
  start: "",
  end: "",
  interval: "daily",
  strategyName: "",
  nSplits: DEFAULT_N_SPLITS,
  testSize: DEFAULT_TEST_SIZE,
  gap: DEFAULT_GAP,
  expanding: true,
};

export interface StrategySchemaParam {
  name: string;
  required: boolean;
}

export interface MissingParamItem {
  loc: string[];
  msg: string;
  type: string;
}

export function findMissingStrategyParams(
  params: readonly StrategySchemaParam[],
  values: Readonly<Record<string, unknown>>,
): MissingParamItem[] {
  return params
    .filter((p) => p.required && values[p.name] === undefined)
    .map((p) => ({
      loc: ["strategy", "params", p.name],
      msg: "field required",
      type: "missing",
    }));
}

export type ExperimentPayload = Record<string, unknown> & {
  name: string;
  seed: number;
  data: {
    source: string;
    tickers: string[];
    start: string;
    end: string;
    interval: string;
  };
  strategy: { name: string; params: Record<string, unknown> };
  validation: { n_splits: number; test_size: number; gap: number; expanding: boolean };
};

/**
 * Convert validated form values + strategy params into the wire payload
 * the backend expects. ``strategyParams`` is owned by the StrategyParamsEditor
 * (each param has its own typed input or JSON editor); we only stitch here.
 */
export function toExperimentPayload(
  values: ConfigureFormValues,
  strategyParams: Record<string, unknown>,
): ExperimentPayload {
  return {
    name: values.name,
    seed: values.seed,
    data: {
      source: "yfinance",
      tickers: parseStringList(values.tickers),
      start: values.start,
      end: values.end,
      interval: values.interval,
    },
    strategy: { name: values.strategyName, params: strategyParams },
    validation: {
      n_splits: values.nSplits,
      test_size: values.testSize,
      gap: values.gap,
      expanding: values.expanding,
    },
  };
}
