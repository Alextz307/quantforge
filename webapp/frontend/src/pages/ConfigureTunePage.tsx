import { useMemo, useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type SubmitHandler } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ExperimentFieldsSection } from "@/components/forms/ExperimentFieldsSection";
import { HpoFieldsSection } from "@/components/forms/HpoFieldsSection";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { SubmitFailureAlert } from "@/components/forms/SubmitFailureAlert";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { buildErrorIndex } from "@/api/errors";
import { useValidateConfig } from "@/api/configs";
import { useStrategySchema } from "@/api/strategies";
import { useStrategyParamsState } from "@/hooks/useStrategyParamsState";
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
  const [strategyParams, setStrategyParams] = useStrategyParamsState(strategyName, schema.data);

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

  return (
    <Card className="max-w-4xl">
      <CardHeader>
        <CardTitle>Configure tune</CardTitle>
        <CardDescription>
          Optuna study over the strategy's <code className="font-mono">suggest_params</code>{" "}
          space. The same experiment fields below are reused inside every trial.
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

          <ServerErrorList errors={serverErrors} />
          <SubmitFailureAlert mutation={submit} />

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
