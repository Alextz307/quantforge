import { z } from "zod";
import { experimentBaseSchema, startBeforeEndRefinement } from "./configureForm";

// Mirrors src.core.hpo_config.{SamplerKind,PrunerKind,ObjectiveKind} StrEnums.
// Server-side validation via POST /api/configs/validate?kind=hpo is the source
// of truth; this list exists for the typed dropdown on the Configure form only.
export const SAMPLER_VALUES = ["tpe", "random", "cmaes", "qmc"] as const;
export const PRUNER_VALUES = ["median", "hyperband", "percentile", "none"] as const;
export const OBJECTIVE_VALUES = ["sharpe", "calmar", "sortino_minus_drawdown"] as const;

export type SamplerValue = (typeof SAMPLER_VALUES)[number];
export type PrunerValue = (typeof PRUNER_VALUES)[number];
export type ObjectiveValue = (typeof OBJECTIVE_VALUES)[number];

// Mirrored from the HPOConfig.study_name backend constraint — imported by
// HpoFieldsSection so the input's HTML maxLength matches the server.
export const STUDY_NAME_MAX = 128;
const DEFAULT_N_TRIALS = 50;
const N_TRIALS_MIN = 1;
const N_TRIALS_MAX = 10_000;
const DEFAULT_N_JOBS = 1;
const N_JOBS_MIN = 1;
const N_JOBS_MAX = 64;
const DEFAULT_HPO_SEED = 42;
const HPO_SEED_MIN = 0;

// Mirrors HPOConfig._validate_study_name (no path separators) so the form
// catches the most common mistake before round-tripping through the backend.
const STUDY_NAME_REGEX = /^[^/\\]+$/;

export const configureTuneFormSchema = experimentBaseSchema
  .extend({
    studyName: z
      .string()
      .min(1, "Study name is required")
      .max(STUDY_NAME_MAX, `Study name must be at most ${String(STUDY_NAME_MAX)} characters`)
      .regex(STUDY_NAME_REGEX, "Study name must not contain path separators"),
    nTrials: z.coerce.number().int().min(N_TRIALS_MIN).max(N_TRIALS_MAX).default(DEFAULT_N_TRIALS),
    nJobs: z.coerce.number().int().min(N_JOBS_MIN).max(N_JOBS_MAX).default(DEFAULT_N_JOBS),
    sampler: z.enum(SAMPLER_VALUES).default("tpe"),
    pruner: z.enum(PRUNER_VALUES).default("median"),
    objective: z.enum(OBJECTIVE_VALUES).default("sharpe"),
    hpoSeed: z.coerce.number().int().min(HPO_SEED_MIN).default(DEFAULT_HPO_SEED),
  })
  .refine(startBeforeEndRefinement.predicate, {
    message: startBeforeEndRefinement.message,
    path: [...startBeforeEndRefinement.path],
  });

export type ConfigureTuneFormValues = z.infer<typeof configureTuneFormSchema>;

export const HPO_FORM_DEFAULTS = {
  studyName: "",
  nTrials: DEFAULT_N_TRIALS,
  nJobs: DEFAULT_N_JOBS,
  sampler: "tpe" as SamplerValue,
  pruner: "median" as PrunerValue,
  objective: "sharpe" as ObjectiveValue,
  hpoSeed: DEFAULT_HPO_SEED,
} as const;

export type HpoPayload = Record<string, unknown> & {
  study_name: string;
  n_trials: number;
  n_jobs: number;
  sampler: SamplerValue;
  pruner: PrunerValue;
  objective: ObjectiveValue;
  seed: number;
};

export function toHpoPayload(values: ConfigureTuneFormValues): HpoPayload {
  return {
    study_name: values.studyName,
    n_trials: values.nTrials,
    n_jobs: values.nJobs,
    sampler: values.sampler,
    pruner: values.pruner,
    objective: values.objective,
    seed: values.hpoSeed,
  };
}
