import { useMemo, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { SubmitFailureAlert } from "@/components/forms/SubmitFailureAlert";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { useRunsPage, type RunSummary } from "@/api/runs";
import { QueryRenderer } from "@/components/QueryRenderer";
import { formatDateTime, formatMetric } from "@/lib/format";
import { jobDetailPath } from "@/lib/routes";

const MIN_RUNS = 2;
const MAX_RUNS = 8;
// Picker fetches a flat page sized for hundreds of runs. Above this count the
// dropdown becomes unusable anyway and the user should narrow via filters first.
const PICKER_PAGE_LIMIT = 200;
const DEFAULT_N_JOBS = 1;
const MAX_N_JOBS = 8;
// Mirror the backend slug pattern so the form rejects bad input before the
// 422 round-trip (still defense-in-depth - the backend re-checks).
const SLUG_PATTERN = /^[A-Za-z0-9_\-:]+$/;

type SignificanceTest = "bootstrap" | "none";

const SIGNIFICANCE_OPTIONS: ReadonlyArray<{ value: SignificanceTest; label: string }> = [
  { value: "bootstrap", label: "Paired Sharpe bootstrap" },
  { value: "none", label: "No significance test" },
];

export function ConfigureComparePage() {
  const navigate = useNavigate();
  const submit = useSubmitJob();
  const runsQuery = useRunsPage({
    limit: PICKER_PAGE_LIMIT,
    offset: 0,
    sortBy: "created_at",
    order: "desc",
  });

  const [selectedIds, setSelectedIds] = useState<readonly string[]>([]);
  const [outName, setOutName] = useState("");
  const [significanceTest, setSignificanceTest] = useState<SignificanceTest>("bootstrap");
  const [nJobs, setNJobs] = useState<number>(DEFAULT_N_JOBS);
  const [writeReport, setWriteReport] = useState(true);
  const [publishLabel, setPublishLabel] = useState("");
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const [clientErrors, setClientErrors] = useState<readonly ValidationErrorItem[]>([]);

  const inlineErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );

  const toggleSelected = (experimentId: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(experimentId)) return prev.filter((id) => id !== experimentId);
      if (prev.length >= MAX_RUNS) return prev;
      return [...prev, experimentId];
    });
  };

  const validate = (): readonly ValidationErrorItem[] => {
    const errors: ValidationErrorItem[] = [];
    if (selectedIds.length < MIN_RUNS) {
      errors.push({
        loc: ["compare_payload", "run_ids"],
        msg: `Pick at least ${String(MIN_RUNS)} runs`,
        type: "value_error",
      });
    }
    if (outName.trim() === "") {
      errors.push({
        loc: ["compare_payload", "out_name"],
        msg: "Output name is required",
        type: "missing",
      });
    } else if (!SLUG_PATTERN.test(outName)) {
      errors.push({
        loc: ["compare_payload", "out_name"],
        msg: "Letters, digits, _ - : only",
        type: "value_error",
      });
    }
    if (publishLabel !== "" && !SLUG_PATTERN.test(publishLabel)) {
      errors.push({
        loc: ["compare_payload", "publish_label"],
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
        kind: "compare",
        compare_payload: {
          run_ids: [...selectedIds],
          out_name: outName,
          significance_test: significanceTest,
          n_jobs: nJobs,
          write_report: writeReport,
          publish_label: publishLabel === "" ? null : publishLabel,
        },
        feature_importance: false,
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
        <CardTitle>Configure comparison</CardTitle>
        <CardDescription>
          Rank 2-8 completed runs head-to-head. Each run's frozen config is reused - no walk-forward
          re-execution.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} noValidate className="space-y-6">
          <section className="space-y-2">
            <div className="flex items-baseline justify-between">
              <Label>
                Runs to compare ({selectedIds.length}/{MAX_RUNS})
              </Label>
              <span className="text-xs text-muted-foreground">
                Minimum {MIN_RUNS}, maximum {MAX_RUNS}
              </span>
            </div>
            <QueryRenderer query={runsQuery} errorTitle="Failed to load runs">
              {(page) => {
                // First selection locks the data_hash: paired-bootstrap rejects
                // mixed bar indices, so disable mismatched rows in the picker
                // rather than letting the CLI reject the comparison post-submit.
                const lockedDataHash =
                  selectedIds.length === 0
                    ? null
                    : (page.items.find((r) => r.experiment_id === selectedIds[0])?.data_hash ??
                      null);
                return (
                  <RunPicker
                    rows={page.items}
                    selectedIds={selectedIds}
                    onToggle={toggleSelected}
                    maxSelected={MAX_RUNS}
                    lockedDataHash={lockedDataHash}
                  />
                );
              }}
            </QueryRenderer>
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col gap-1">
              <Label htmlFor="compare-out-name">Output name</Label>
              <Input
                id="compare-out-name"
                value={outName}
                placeholder="e.g. ab_vs_vt_q3"
                onChange={(e) => {
                  setOutName(e.target.value);
                }}
              />
              <p className="text-xs text-muted-foreground">
                Directory under <code className="font-mono">experiment_results/comparisons/</code>.
              </p>
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="compare-significance">Significance test</Label>
              <select
                id="compare-significance"
                className="h-10 rounded-md border border-input bg-background px-3 text-sm"
                value={significanceTest}
                onChange={(e) => {
                  setSignificanceTest(e.target.value as SignificanceTest);
                }}
              >
                {SIGNIFICANCE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="compare-n-jobs">Parallel workers</Label>
              <Input
                id="compare-n-jobs"
                type="number"
                min={1}
                max={MAX_N_JOBS}
                value={nJobs}
                onChange={(e) => {
                  const parsed = Number.parseInt(e.target.value, 10);
                  setNJobs(Number.isFinite(parsed) ? parsed : DEFAULT_N_JOBS);
                }}
              />
              <p className="text-xs text-muted-foreground">
                Reuse mode is in-process for n_jobs=1; &gt;1 fans out via ProcessPoolExecutor.
              </p>
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="compare-publish-label">Publish label (optional)</Label>
              <Input
                id="compare-publish-label"
                value={publishLabel}
                placeholder="e.g. tab:ab_vs_vt"
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
            Write ranking + significance LaTeX tables + equity overlay plot
          </label>

          <ServerErrorList errors={inlineErrors} />
          <SubmitFailureAlert mutation={submit} />

          <div className="flex justify-end gap-2">
            <Button type="submit" disabled={submit.isPending}>
              {submit.isPending ? "Launching..." : "Launch comparison"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

interface RunPickerProps {
  rows: readonly RunSummary[];
  selectedIds: readonly string[];
  onToggle: (experimentId: string) => void;
  maxSelected: number;
  lockedDataHash: string | null;
}

function RunPicker({ rows, selectedIds, onToggle, maxSelected, lockedDataHash }: RunPickerProps) {
  if (rows.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No completed runs yet. Launch a run first under{" "}
        <code className="font-mono">/configure/run</code>.
      </p>
    );
  }
  const selectedSet = new Set(selectedIds);
  return (
    <div className="space-y-1">
      {lockedDataHash !== null && (
        <p className="text-xs text-muted-foreground" data-testid="compare-data-hash-lock-notice">
          Selection locked to runs sharing the first pick's bar series (tickers + date range +
          interval). The paired bootstrap requires identical fold indices.
        </p>
      )}
      <div className="max-h-72 overflow-y-auto rounded-md border" data-testid="compare-run-picker">
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
            {rows.map((r) => {
              const checked = selectedSet.has(r.experiment_id);
              const dataMismatch =
                lockedDataHash !== null && !checked && r.data_hash !== lockedDataHash;
              const capReached = !checked && selectedIds.length >= maxSelected;
              const disabled = capReached || dataMismatch;
              return (
                <tr
                  key={r.experiment_id}
                  className={`border-t last:border-b-0 ${dataMismatch ? "opacity-50" : ""}`}
                  data-testid={`compare-run-row-${r.experiment_id}`}
                >
                  <td className="py-2 pl-3">
                    <input
                      type="checkbox"
                      aria-label={`Select ${r.name}`}
                      checked={checked}
                      disabled={disabled}
                      title={dataMismatch ? "Different bar series than first selection" : undefined}
                      onChange={() => {
                        onToggle(r.experiment_id);
                      }}
                    />
                  </td>
                  <td className="py-2 pr-4">{r.name}</td>
                  <td className="py-2 pr-4 font-mono">{r.strategy}</td>
                  <td className="py-2 pr-4 font-mono">{r.tickers.join(", ")}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{formatDateTime(r.created_at)}</td>
                  <td className="py-2 pr-4 text-right font-mono">
                    {formatMetric(r.sharpe_mean, 3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
