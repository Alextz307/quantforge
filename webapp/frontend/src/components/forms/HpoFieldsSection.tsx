import type { FieldErrors, UseFormRegister } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { ConfigField } from "@/components/forms/ConfigField";
import {
  PRUNER_VALUES,
  SAMPLER_VALUES,
  STUDY_NAME_MAX,
  type ConfigureTuneFormValues,
} from "@/lib/schemas/configureTuneForm";

const SELECT_CLASS =
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm";

export interface HpoFieldsSectionProps {
  register: UseFormRegister<ConfigureTuneFormValues>;
  errors: FieldErrors<ConfigureTuneFormValues>;
  isSubmitting: boolean;
}

export function HpoFieldsSection({ register, errors, isSubmitting }: HpoFieldsSectionProps) {
  return (
    <>
      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <ConfigField
          id="studyName"
          label="Study name"
          hint="Doubles as the directory under experiment_results/hpo/"
          error={errors.studyName?.message}
          className="md:col-span-2"
        >
          <Input
            id="studyName"
            maxLength={STUDY_NAME_MAX}
            {...register("studyName")}
            disabled={isSubmitting}
          />
        </ConfigField>
        <ConfigField id="nTrials" label="n_trials" error={errors.nTrials?.message}>
          <Input id="nTrials" type="number" {...register("nTrials")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="nJobs" label="n_jobs" error={errors.nJobs?.message}>
          <Input id="nJobs" type="number" {...register("nJobs")} disabled={isSubmitting} />
        </ConfigField>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ConfigField id="sampler" label="Sampler" error={errors.sampler?.message}>
          <select
            id="sampler"
            {...register("sampler")}
            disabled={isSubmitting}
            className={SELECT_CLASS}
          >
            {SAMPLER_VALUES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </ConfigField>
        <ConfigField id="pruner" label="Pruner" error={errors.pruner?.message}>
          <select
            id="pruner"
            {...register("pruner")}
            disabled={isSubmitting}
            className={SELECT_CLASS}
          >
            {PRUNER_VALUES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </ConfigField>
        <ConfigField id="hpoSeed" label="HPO seed" error={errors.hpoSeed?.message}>
          <Input id="hpoSeed" type="number" {...register("hpoSeed")} disabled={isSubmitting} />
        </ConfigField>
      </section>
    </>
  );
}
