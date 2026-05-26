import { useMemo, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { SubmitFailureAlert } from "@/components/forms/SubmitFailureAlert";
import { QueryRenderer } from "@/components/QueryRenderer";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { useConfigList, type ConfigEntry } from "@/api/configs";
import { jobDetailPath } from "@/lib/routes";

function parseLegList(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

export function ConfigureStudyPage() {
  const navigate = useNavigate();
  const submit = useSubmitJob();
  const specsQuery = useConfigList("study");

  const [specName, setSpecName] = useState("");
  const [forceRerun, setForceRerun] = useState(false);
  const [skipCompares, setSkipCompares] = useState(false);
  const [skipHoldoutEval, setSkipHoldoutEval] = useState(false);
  const [onlyLegsRaw, setOnlyLegsRaw] = useState("");
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const [clientErrors, setClientErrors] = useState<readonly ValidationErrorItem[]>([]);

  const inlineErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );

  const validate = (): readonly ValidationErrorItem[] => {
    const errors: ValidationErrorItem[] = [];
    if (specName.trim() === "") {
      errors.push({
        loc: ["study_payload", "spec_name"],
        msg: "Pick a study spec",
        type: "missing",
      });
    }
    return errors;
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setServerErrors([]);
    const local = validate();
    if (local.length > 0) {
      setClientErrors(local);
      return;
    }
    setClientErrors([]);
    try {
      const job = await submit.mutateAsync({
        kind: "study",
        study_payload: {
          spec_name: specName,
          force_rerun: forceRerun,
          only_legs: parseLegList(onlyLegsRaw),
          skip_compares: skipCompares,
          skip_holdout_eval: skipHoldoutEval,
        },
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
    <Card className="max-w-3xl">
      <CardHeader>
        <CardTitle>Configure study</CardTitle>
        <CardDescription>
          Pick a spec from <code className="font-mono">config/study/</code>. The orchestrator
          cross-products it into strategy × universe legs and runs tune → walk-forward → holdout for
          each. Live leg grid updates as legs land.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} noValidate className="space-y-6">
          <section className="space-y-2">
            <Label htmlFor="study-spec">Spec</Label>
            <QueryRenderer query={specsQuery} errorTitle="Failed to load study specs">
              {(specs) => <SpecPicker specs={specs} value={specName} onChange={setSpecName} />}
            </QueryRenderer>
          </section>

          <div className="flex flex-col gap-2">
            <Label htmlFor="study-only-legs">Only legs (optional)</Label>
            <Input
              id="study-only-legs"
              value={onlyLegsRaw}
              placeholder="e.g. AdaptiveBollinger__spy_daily_5y, MomentumGatekeeper__qqq_daily_5y"
              onChange={(e) => {
                setOnlyLegsRaw(e.target.value);
              }}
            />
            <p className="text-xs text-muted-foreground">
              Comma-separated leg ids (<code className="font-mono">strategy__universe</code>). Leave
              blank to run every leg in the spec.
            </p>
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Flags</Label>
            <div className="flex flex-col gap-2 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={forceRerun}
                  onChange={(e) => {
                    setForceRerun(e.target.checked);
                  }}
                />
                Force rerun — ignore <code className="font-mono">is_complete</code> markers and
                re-run every leg from scratch
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={skipCompares}
                  onChange={(e) => {
                    setSkipCompares(e.target.checked);
                  }}
                />
                Skip per-universe cross-strategy compares
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={skipHoldoutEval}
                  onChange={(e) => {
                    setSkipHoldoutEval(e.target.checked);
                  }}
                />
                Skip holdout-eval on every leg
              </label>
            </div>
          </div>

          <ServerErrorList errors={inlineErrors} />
          <SubmitFailureAlert mutation={submit} />

          <div className="flex justify-end gap-2">
            <Button type="submit" disabled={submit.isPending}>
              {submit.isPending ? "Launching…" : "Launch study"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

interface SpecPickerProps {
  specs: readonly ConfigEntry[];
  value: string;
  onChange: (name: string) => void;
}

function SpecPicker({ specs, value, onChange }: SpecPickerProps) {
  if (specs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No study specs found under <code className="font-mono">config/study/</code>. Add one and
        reload.
      </p>
    );
  }
  return (
    <select
      id="study-spec"
      className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
      value={value}
      onChange={(e) => {
        onChange(e.target.value);
      }}
    >
      <option value="">— pick a spec —</option>
      {specs.map((s) => (
        <option key={s.name} value={s.name}>
          {s.name}
        </option>
      ))}
    </select>
  );
}
