import type { FieldErrors, UseFormRegister, UseFormSetValue } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { ConfigField } from "@/components/forms/ConfigField";
import { StrategyParamsEditor } from "@/components/forms/StrategyParamsEditor";
import { UniversePicker, type UniversePreset } from "@/components/forms/UniversePicker";
import { useStrategies, type StrategySchema } from "@/api/strategies";
import {
  INTERVAL_OPTIONS,
  TICKERS_INPUT_HINT,
  type ConfigureFormValues,
  type IntervalValue,
} from "@/lib/schemas/configureForm";

/**
 * The form fields shared between Run and Tune launchers (everything in
 * ``ExperimentConfig`` — data block, strategy, walk-forward knobs). Tune
 * adds its own ``<HpoFieldsSection>`` underneath; Run uses this block
 * standalone.
 *
 * Generic over the form's value type so a Tune form (superset of run
 * fields) can pass its own ``UseFormReturn`` without casts.
 */
export interface ExperimentFieldsSectionProps<T extends ConfigureFormValues> {
  register: UseFormRegister<T>;
  setValue: UseFormSetValue<T>;
  errors: FieldErrors<T>;
  strategyParams: Record<string, unknown>;
  onStrategyParamsChange: (next: Record<string, unknown>) => void;
  schemaData: StrategySchema | undefined;
  errorsByLoc: ReadonlyMap<string, string>;
  isSubmitting: boolean;
}

export function ExperimentFieldsSection<T extends ConfigureFormValues>({
  register,
  setValue,
  errors,
  strategyParams,
  onStrategyParamsChange,
  schemaData,
  errorsByLoc,
  isSubmitting,
}: ExperimentFieldsSectionProps<T>) {
  const strategies = useStrategies();

  const applyUniverse = (preset: UniversePreset) => {
    type Path = Parameters<typeof setValue>[0];
    setValue("tickers" as Path, preset.tickers as never, { shouldDirty: true });
    setValue("start" as Path, preset.start as never, { shouldDirty: true });
    setValue("end" as Path, preset.end as never, { shouldDirty: true });
    setValue("interval" as Path, preset.interval as never, { shouldDirty: true });
  };

  // RHF's `errors` on a generic `T` types each field as `FieldError | undefined`,
  // and `register` paths similarly need narrowing — single cast at the boundary
  // keeps the JSX free of repeated `as never` annotations.
  const e = errors as FieldErrors<ConfigureFormValues>;
  const reg = register as unknown as UseFormRegister<ConfigureFormValues>;

  return (
    <>
      <UniversePicker onApply={applyUniverse} disabled={isSubmitting} />

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <ConfigField id="name" label="Run name" error={e.name?.message}>
          <Input id="name" {...reg("name")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="seed" label="Seed" error={e.seed?.message}>
          <Input id="seed" type="number" {...reg("seed")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField
          id="tickers"
          label="Tickers"
          hint={TICKERS_INPUT_HINT}
          error={e.tickers?.message}
          className="md:col-span-2"
        >
          <Input id="tickers" {...reg("tickers")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="start" label="Start" error={e.start?.message}>
          <Input id="start" type="date" {...reg("start")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="end" label="End" error={e.end?.message}>
          <Input id="end" type="date" {...reg("end")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="interval" label="Interval" error={e.interval?.message}>
          <select
            id="interval"
            {...reg("interval")}
            disabled={isSubmitting}
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            {INTERVAL_OPTIONS.map((iv: IntervalValue) => (
              <option key={iv} value={iv}>
                {iv}
              </option>
            ))}
          </select>
        </ConfigField>
      </section>

      <section className="space-y-4">
        <ConfigField id="strategyName" label="Strategy" error={e.strategyName?.message}>
          <select
            id="strategyName"
            {...reg("strategyName")}
            disabled={isSubmitting || strategies.isLoading}
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            <option value="">— select strategy —</option>
            {strategies.data?.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </select>
        </ConfigField>

        {schemaData && (
          <StrategyParamsEditor
            schema={schemaData}
            values={strategyParams}
            onChange={onStrategyParamsChange}
            errorsByLoc={errorsByLoc}
            disabled={isSubmitting}
          />
        )}
      </section>

      <section className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <ConfigField id="nSplits" label="n_splits" error={e.nSplits?.message}>
          <Input id="nSplits" type="number" {...reg("nSplits")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="testSize" label="test_size" error={e.testSize?.message}>
          <Input id="testSize" type="number" {...reg("testSize")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="gap" label="gap" error={e.gap?.message}>
          <Input id="gap" type="number" {...reg("gap")} disabled={isSubmitting} />
        </ConfigField>
        <ConfigField id="expanding" label="expanding" error={e.expanding?.message}>
          <input
            id="expanding"
            type="checkbox"
            className="h-5 w-5"
            {...reg("expanding")}
            disabled={isSubmitting}
          />
        </ConfigField>
      </section>
    </>
  );
}
