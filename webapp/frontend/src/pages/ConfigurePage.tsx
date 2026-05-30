import { useMemo, useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type SubmitHandler } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ExperimentFieldsSection } from "@/components/forms/ExperimentFieldsSection";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { SubmitFailureAlert } from "@/components/forms/SubmitFailureAlert";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { buildErrorIndex } from "@/api/errors";
import { useValidateConfig } from "@/api/configs";
import { useStrategySchema } from "@/api/strategies";
import { useStrategyParamsState } from "@/hooks/useStrategyParamsState";
import { jobDetailPath } from "@/lib/routes";
import {
  configureFormSchema,
  EXPERIMENT_FORM_DEFAULTS,
  findMissingStrategyParams,
  toExperimentPayload,
  type ConfigureFormValues,
} from "@/lib/schemas/configureForm";

export function ConfigurePage() {
  const navigate = useNavigate();

  const form = useForm<ConfigureFormValues>({
    resolver: zodResolver(configureFormSchema),
    defaultValues: EXPERIMENT_FORM_DEFAULTS,
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

  const onSubmit: SubmitHandler<ConfigureFormValues> = async (values) => {
    setServerErrors([]);
    if (schema.data) {
      const missing = findMissingStrategyParams(schema.data.params, strategyParams);
      if (missing.length > 0) {
        setServerErrors(missing);
        return;
      }
    }
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

          <ServerErrorList errors={serverErrors} />
          <SubmitFailureAlert mutation={submit} />

          <div className="flex justify-end gap-2">
            <Button type="submit" disabled={isSubmitting || submit.isPending}>
              {submit.isPending ? "Launching..." : "Launch run"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
