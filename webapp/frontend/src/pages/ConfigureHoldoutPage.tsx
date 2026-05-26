import { useMemo, useState, type FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { SubmitFailureAlert } from "@/components/forms/SubmitFailureAlert";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { useHpoStudies, type HpoSummary } from "@/api/hpo";
import { useRunsPage, type RunSummary } from "@/api/runs";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, formatMetric } from "@/lib/format";
import { jobDetailPath } from "@/lib/routes";
import { SOURCE_KIND_HPO, SOURCE_KIND_RUN, type SourceKind } from "@/lib/sourceKind";

const PICKER_PAGE_LIMIT = 200;
const SLUG_PATTERN = /^[A-Za-z0-9_\-:]+$/;
const QUERY_SOURCE_KIND = "source_kind";
const QUERY_SOURCE_ID = "source_id";

function isSourceKind(value: string | null): value is SourceKind {
  return value === SOURCE_KIND_RUN || value === SOURCE_KIND_HPO;
}

export function ConfigureHoldoutPage() {
  const navigate = useNavigate();
  const submit = useSubmitJob();
  const [searchParams] = useSearchParams();

  const initialSourceKind: SourceKind = isSourceKind(searchParams.get(QUERY_SOURCE_KIND))
    ? (searchParams.get(QUERY_SOURCE_KIND) as SourceKind)
    : SOURCE_KIND_RUN;
  const initialSourceId = searchParams.get(QUERY_SOURCE_ID) ?? "";

  const [sourceKind, setSourceKind] = useState<SourceKind>(initialSourceKind);
  const [sourceId, setSourceId] = useState<string>(initialSourceId);
  const [outName, setOutName] = useState("");
  const [writeReport, setWriteReport] = useState(true);
  const [publishLabel, setPublishLabel] = useState("");
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const [clientErrors, setClientErrors] = useState<readonly ValidationErrorItem[]>([]);

  const runsQuery = useRunsPage({
    limit: PICKER_PAGE_LIMIT,
    offset: 0,
    sortBy: "created_at",
    order: "desc",
  });
  const hpoQuery = useHpoStudies();

  const inlineErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );

  const validate = (): readonly ValidationErrorItem[] => {
    const errors: ValidationErrorItem[] = [];
    if (sourceId.trim() === "") {
      errors.push({
        loc: ["holdout_payload", "source_id"],
        msg: "Pick a source",
        type: "missing",
      });
    }
    if (outName !== "" && !SLUG_PATTERN.test(outName)) {
      errors.push({
        loc: ["holdout_payload", "out_name"],
        msg: "Letters, digits, _ - : only",
        type: "value_error",
      });
    }
    if (publishLabel !== "" && !SLUG_PATTERN.test(publishLabel)) {
      errors.push({
        loc: ["holdout_payload", "publish_label"],
        msg: "Letters, digits, _ - : only",
        type: "value_error",
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
        kind: "holdout",
        holdout_payload: {
          source_kind: sourceKind,
          source_id: sourceId,
          out_name: outName === "" ? null : outName,
          write_report: writeReport,
          publish_label: publishLabel === "" ? null : publishLabel,
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
    <Card className="max-w-4xl">
      <CardHeader>
        <CardTitle>Configure holdout eval</CardTitle>
        <CardDescription>
          Refit the source's strategy on the full dev region, then evaluate once on the reserved
          holdout. Mutually exclusive sources: a completed run with a holdout boundary, OR an HPO
          study's best config.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} noValidate className="space-y-6">
          <fieldset className="space-y-2">
            <legend className="text-sm font-medium">Source kind</legend>
            <div className="flex flex-col gap-2 sm:flex-row sm:gap-6">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="source-kind"
                  value={SOURCE_KIND_RUN}
                  checked={sourceKind === SOURCE_KIND_RUN}
                  onChange={() => {
                    setSourceKind(SOURCE_KIND_RUN);
                    setSourceId("");
                  }}
                />
                From completed run
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="source-kind"
                  value={SOURCE_KIND_HPO}
                  checked={sourceKind === SOURCE_KIND_HPO}
                  onChange={() => {
                    setSourceKind(SOURCE_KIND_HPO);
                    setSourceId("");
                  }}
                />
                From HPO best config
              </label>
            </div>
          </fieldset>

          {sourceKind === SOURCE_KIND_RUN ? (
            <QueryRenderer query={runsQuery} errorTitle="Failed to load runs">
              {(page) => (
                <RunSourcePicker rows={page.items} selectedId={sourceId} onSelect={setSourceId} />
              )}
            </QueryRenderer>
          ) : (
            <QueryRenderer query={hpoQuery} errorTitle="Failed to load HPO studies">
              {(rows) => (
                <HpoSourcePicker rows={rows} selectedId={sourceId} onSelect={setSourceId} />
              )}
            </QueryRenderer>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col gap-1">
              <Label htmlFor="holdout-out-name">Output name (optional)</Label>
              <Input
                id="holdout-out-name"
                value={outName}
                placeholder="defaults to source basename"
                onChange={(e) => {
                  setOutName(e.target.value);
                }}
              />
              <p className="text-xs text-muted-foreground">
                Directory under <code className="font-mono">experiment_results/holdout_evals/</code>
                .
              </p>
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="holdout-publish-label">Publish label (optional)</Label>
              <Input
                id="holdout-publish-label"
                value={publishLabel}
                placeholder="e.g. tab:vt_holdout"
                onChange={(e) => {
                  setPublishLabel(e.target.value);
                }}
              />
              <p className="text-xs text-muted-foreground">
                Stable LaTeX caption/label slug for re-runs.
              </p>
            </div>
          </div>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={writeReport}
              onChange={(e) => {
                setWriteReport(e.target.checked);
              }}
            />
            Write holdout-metrics LaTeX table + holdout-equity plot
          </label>

          <ServerErrorList errors={inlineErrors} />
          <SubmitFailureAlert mutation={submit} />

          <div className="flex justify-end gap-2">
            <Button type="submit" disabled={submit.isPending}>
              {submit.isPending ? "Launching…" : "Launch holdout"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

interface RunSourcePickerProps {
  rows: readonly RunSummary[];
  selectedId: string;
  onSelect: (experimentId: string) => void;
}

function RunSourcePicker({ rows, selectedId, onSelect }: RunSourcePickerProps) {
  const eligible = useMemo(() => rows.filter((r) => r.has_holdout), [rows]);
  if (eligible.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No runs with a holdout boundary on disk. Add{" "}
        <code className="font-mono">holdout_start</code> to your experiment config and re-run.
      </p>
    );
  }
  return (
    <div className="max-h-72 overflow-y-auto rounded-md border" data-testid="holdout-run-picker">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="w-8 py-2 pl-3" />
            <th className="py-2 pr-4">Name</th>
            <th className="py-2 pr-4 font-mono">Strategy</th>
            <th className="py-2 pr-4 font-mono">Tickers</th>
            <th className="py-2 pr-4 font-mono">Created</th>
            <th className="py-2 pr-4 text-right font-mono">Sharpe</th>
          </tr>
        </thead>
        <tbody>
          {eligible.map((r) => (
            <tr
              key={r.experiment_id}
              className="border-t last:border-b-0"
              data-testid={`holdout-run-row-${r.experiment_id}`}
            >
              <td className="py-2 pl-3">
                <input
                  type="radio"
                  name="holdout-run-source"
                  aria-label={`Select ${r.name}`}
                  checked={selectedId === r.experiment_id}
                  onChange={() => {
                    onSelect(r.experiment_id);
                  }}
                />
              </td>
              <td className="py-2 pr-4">{r.name}</td>
              <td className="py-2 pr-4 font-mono">{r.strategy}</td>
              <td className="py-2 pr-4 font-mono">{r.tickers.join(", ")}</td>
              <td className="py-2 pr-4 font-mono text-xs">{formatDateTime(r.created_at)}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatMetric(r.sharpe_mean, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface HpoSourcePickerProps {
  rows: readonly HpoSummary[];
  selectedId: string;
  onSelect: (studyName: string) => void;
}

function HpoSourcePicker({ rows, selectedId, onSelect }: HpoSourcePickerProps) {
  // Filter on both flags: best_config.yaml must exist AND its validation block
  // must reserve a holdout region. The CLI rejects holdout-eval otherwise.
  const eligible = useMemo(
    () => rows.filter((r) => r.has_best_config && r.best_config_reserves_holdout),
    [rows],
  );
  if (eligible.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No HPO studies with a holdout-eligible <code className="font-mono">best_config.yaml</code>.
        Need a completed study whose <code className="font-mono">validation.holdout_pct</code> is
        non-zero, or pick the run-source variant.
      </p>
    );
  }
  return (
    <div className="max-h-72 overflow-y-auto rounded-md border" data-testid="holdout-hpo-picker">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="w-8 py-2 pl-3" />
            <th className="py-2 pr-4">Name</th>
            <th className="py-2 pr-4 font-mono">Trials</th>
            <th className="py-2 pr-4 font-mono">Completed</th>
            <th className="py-2 pr-4 text-right font-mono">Best value</th>
            <th className="py-2 pr-4 font-mono text-xs">Created</th>
          </tr>
        </thead>
        <tbody>
          {eligible.map((r) => (
            <tr
              key={r.wire_id}
              className="border-t last:border-b-0"
              data-testid={`holdout-hpo-row-${r.wire_id}`}
            >
              <td className="py-2 pl-3">
                <input
                  type="radio"
                  name="holdout-hpo-source"
                  aria-label={`Select ${r.name}`}
                  checked={selectedId === r.wire_id}
                  onChange={() => {
                    onSelect(r.wire_id);
                  }}
                />
              </td>
              <td className="py-2 pr-4">{r.name}</td>
              <td className="py-2 pr-4 font-mono">{r.n_trials}</td>
              <td className="py-2 pr-4 font-mono">{r.n_complete}</td>
              <td className="py-2 pr-4 text-right font-mono">{formatMetric(r.best_value, 3)}</td>
              <td className="py-2 pr-4 font-mono text-xs">{formatDateTime(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
