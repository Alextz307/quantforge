import { useEffect, useMemo, useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type SubmitHandler } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ExperimentFieldsSection } from "@/components/forms/ExperimentFieldsSection";
import { HpoFieldsSection } from "@/components/forms/HpoFieldsSection";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { buildErrorIndex } from "@/api/errors";
import { useValidateConfig } from "@/api/configs";
import { usePublicSettings } from "@/api/settings";
import { useStrategySchema } from "@/api/strategies";
import { jobDetailPath } from "@/lib/routes";
import {
  EXPERIMENT_FORM_DEFAULTS,
  findMissingStrategyParams,
  toExperimentPayload,
} from "@/lib/schemas/configureForm";
import {
  configureTuneFormSchema,
  HPO_FORM_DEFAULTS,
  toHpoPayload,
  type ConfigureTuneFormValues,
} from "@/lib/schemas/configureTuneForm";

const DEFAULT_FORM_VALUES: ConfigureTuneFormValues = {
  ...EXPERIMENT_FORM_DEFAULTS,
  ...HPO_FORM_DEFAULTS,
};

export function ConfigureTunePage() {
  const navigate = useNavigate();
  const settings = usePublicSettings();

  const form = useForm<ConfigureTuneFormValues>({
    resolver: zodResolver(configureTuneFormSchema),
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
  const schemaQualname = schema.data?.qualname;
  useEffect(() => {
    setStrategyParams({ ...(schema.data?.canonical_params ?? {}) });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- gated on qualname intentionally
  }, [strategyName, schemaQualname]);

  const validate = useValidateConfig();
  const submit = useSubmitJob();
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const errorsByLoc = useMemo(() => buildErrorIndex(serverErrors), [serverErrors]);

  const onSubmit: SubmitHandler<ConfigureTuneFormValues> = async (values) => {
    setServerErrors([]);
    if (schema.data) {
      const missing = findMissingStrategyParams(schema.data.params, strategyParams);
      if (missing.length > 0) {
        setServerErrors(missing);
        return;
      }
    }
    const experimentPayload = toExperimentPayload(values, strategyParams);
    const hpoPayload = toHpoPayload(values);

    // Validate experiment + hpo in parallel; surface experiment errors first
    // (users edit the experiment block more often than the tune knobs).
    const [expValidation, hpoValidation] = await Promise.all([
      validate.mutateAsync({ kind: "experiment", payload: experimentPayload }),
      validate.mutateAsync({ kind: "hpo", payload: hpoPayload }),
    ]);
    if (!expValidation.valid) {
      setServerErrors(expValidation.errors);
      return;
    }
    if (!hpoValidation.valid) {
      setServerErrors(hpoValidation.errors.map((e) => ({ ...e, loc: ["hpo_payload", ...e.loc] })));
      return;
    }

    try {
      const job = await submit.mutateAsync({
        kind: "tune",
        config_payload: experimentPayload,
        hpo_payload: hpoPayload,
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
          enable launching new tune studies from the UI.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Card className="max-w-4xl">
      <CardHeader>
        <CardTitle>Configure tune</CardTitle>
        <CardDescription>
          Optuna study over the strategy's <code className="font-mono">suggest_params</code> space.
          The same experiment fields below are reused inside every trial.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} noValidate className="space-y-6">
          <ExperimentFieldsSection
            register={register}
            setValue={setValue}
            errors={errors}
            strategyParams={strategyParams}
            onStrategyParamsChange={setStrategyParams}
            schemaData={schema.data}
            errorsByLoc={errorsByLoc}
            isSubmitting={isSubmitting}
          />

          <HpoFieldsSection register={register} errors={errors} isSubmitting={isSubmitting} />

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
              {submit.isPending ? "Launching…" : "Launch tune"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
