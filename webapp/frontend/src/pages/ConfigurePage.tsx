import { useEffect, useMemo, useState, type ReactNode } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type SubmitHandler } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StrategyParamsEditor } from "@/components/forms/StrategyParamsEditor";
import { UniversePicker, type UniversePreset } from "@/components/forms/UniversePicker";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { useValidateConfig } from "@/api/configs";
import { usePublicSettings } from "@/api/settings";
import { useStrategies, useStrategySchema } from "@/api/strategies";
import { jobDetailPath } from "@/lib/routes";
import {
  configureFormSchema,
  toExperimentPayload,
  TICKERS_INPUT_HINT,
  INTERVAL_OPTIONS,
  type ConfigureFormValues,
  type IntervalValue,
} from "@/lib/schemas/configureForm";

const DEFAULT_FORM_VALUES: ConfigureFormValues = {
  name: "",
  seed: 42,
  tickers: "",
  start: "",
  end: "",
  interval: "daily",
  strategyName: "",
  nSplits: 3,
  testSize: 252,
  gap: 5,
  expanding: true,
};

export function ConfigurePage() {
  const navigate = useNavigate();
  const settings = usePublicSettings();
  const strategies = useStrategies();

  const form = useForm<ConfigureFormValues>({
    resolver: zodResolver(configureFormSchema),
    defaultValues: DEFAULT_FORM_VALUES,
  });
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors, isSubmitting },
  } = form;

  const strategyName = watch("strategyName");
  const schema = useStrategySchema(strategyName || null);
  const [strategyParams, setStrategyParams] = useState<Record<string, unknown>>({});
  // Reset params when the strategy choice changes — values from the previous
  // strategy aren't meaningful and would silently leak through to the payload.
  useEffect(() => {
    setStrategyParams({});
  }, [strategyName]);

  const validate = useValidateConfig();
  const submit = useSubmitJob();
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const errorsByLoc = useMemo(() => buildErrorIndex(serverErrors), [serverErrors]);

  const applyUniverse = (preset: UniversePreset) => {
    setValue("tickers", preset.tickers, { shouldDirty: true });
    setValue("start", preset.start, { shouldDirty: true });
    setValue("end", preset.end, { shouldDirty: true });
    setValue("interval", preset.interval, { shouldDirty: true });
  };

  const onSubmit: SubmitHandler<ConfigureFormValues> = async (values) => {
    setServerErrors([]);
    const payload = toExperimentPayload(values, strategyParams);
    const validation = await validate.mutateAsync({ kind: "experiment", payload });
    if (!validation.valid) {
      setServerErrors(validation.errors);
      return;
    }
    try {
      const job = await submit.mutateAsync({
        kind: "run",
        config_payload: payload,
        overrides: [],
      });
      navigate(jobDetailPath(job.id));
    } catch (err) {
      if (err instanceof SubmitJobError) {
        setServerErrors(err.fieldErrors);
        return;
      }
      throw err;
    }
  };

  if (settings.isLoading) {
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  }
  if (settings.data && !settings.data.jobs_enabled) {
    return (
      <Alert variant="destructive" className="max-w-2xl">
        <AlertTitle>Job execution is disabled</AlertTitle>
        <AlertDescription>
          Set <code className="font-mono">WEBAPP_JOBS_ENABLED=true</code> and restart the backend to
          enable launching new runs from the UI.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Card className="max-w-4xl">
      <CardHeader>
        <CardTitle>Configure run</CardTitle>
        <CardDescription>
          Build an experiment config and launch it as a background job. Server-side validation
          mirrors the YAML rules used by the CLI.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} noValidate className="space-y-6">
          <UniversePicker onApply={applyUniverse} disabled={isSubmitting} />

          <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Field id="name" label="Run name" error={errors.name?.message}>
              <Input id="name" {...register("name")} disabled={isSubmitting} />
            </Field>
            <Field id="seed" label="Seed" error={errors.seed?.message}>
              <Input id="seed" type="number" {...register("seed")} disabled={isSubmitting} />
            </Field>
            <Field
              id="tickers"
              label="Tickers"
              hint={TICKERS_INPUT_HINT}
              error={errors.tickers?.message}
              className="md:col-span-2"
            >
              <Input id="tickers" {...register("tickers")} disabled={isSubmitting} />
            </Field>
            <Field id="start" label="Start" error={errors.start?.message}>
              <Input id="start" type="date" {...register("start")} disabled={isSubmitting} />
            </Field>
            <Field id="end" label="End" error={errors.end?.message}>
              <Input id="end" type="date" {...register("end")} disabled={isSubmitting} />
            </Field>
            <Field id="interval" label="Interval" error={errors.interval?.message}>
              <select
                id="interval"
                {...register("interval")}
                disabled={isSubmitting}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                {INTERVAL_OPTIONS.map((iv: IntervalValue) => (
                  <option key={iv} value={iv}>
                    {iv}
                  </option>
                ))}
              </select>
            </Field>
          </section>

          <section className="space-y-4">
            <Field id="strategyName" label="Strategy" error={errors.strategyName?.message}>
              <select
                id="strategyName"
                {...register("strategyName")}
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
            </Field>

            {schema.data && (
              <StrategyParamsEditor
                schema={schema.data}
                values={strategyParams}
                onChange={setStrategyParams}
                errorsByLoc={errorsByLoc}
                disabled={isSubmitting}
              />
            )}
          </section>

          <section className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Field id="nSplits" label="n_splits" error={errors.nSplits?.message}>
              <Input id="nSplits" type="number" {...register("nSplits")} disabled={isSubmitting} />
            </Field>
            <Field id="testSize" label="test_size" error={errors.testSize?.message}>
              <Input
                id="testSize"
                type="number"
                {...register("testSize")}
                disabled={isSubmitting}
              />
            </Field>
            <Field id="gap" label="gap" error={errors.gap?.message}>
              <Input id="gap" type="number" {...register("gap")} disabled={isSubmitting} />
            </Field>
            <Field id="expanding" label="expanding" error={errors.expanding?.message}>
              <input
                id="expanding"
                type="checkbox"
                className="h-5 w-5"
                {...register("expanding")}
                disabled={isSubmitting}
              />
            </Field>
          </section>

          {serverErrors.length > 0 && (
            <Alert variant="destructive">
              <AlertTitle>Backend rejected the config</AlertTitle>
              <AlertDescription>
                <ul className="list-disc space-y-1 pl-4">
                  {serverErrors.map((err, idx) => (
                    <li key={idx} className="font-mono text-xs">
                      <strong>{err.loc.join(".")}</strong>: {err.msg}
                    </li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {submit.isError && !(submit.error instanceof SubmitJobError) && (
            <Alert variant="destructive">
              <AlertDescription>{submit.error.message}</AlertDescription>
            </Alert>
          )}

          <div className="flex justify-end gap-2">
            <Button type="submit" disabled={isSubmitting || submit.isPending}>
              {submit.isPending ? "Launching…" : "Launch run"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

function buildErrorIndex(errors: readonly ValidationErrorItem[]): ReadonlyMap<string, string> {
  const out = new Map<string, string>();
  for (const err of errors) out.set(err.loc.join("."), err.msg);
  return out;
}

interface FieldProps {
  id: string;
  label: string;
  hint?: string | undefined;
  error?: string | undefined;
  className?: string | undefined;
  children: ReactNode;
}

function Field({ id, label, hint, error, className, children }: FieldProps) {
  return (
    <div className={className}>
      <Label htmlFor={id} className="mb-1.5 block">
        {label}
      </Label>
      {children}
      {hint && !error && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
      {error && <p className="mt-1 text-xs text-rose-600">{error}</p>}
    </div>
  );
}
