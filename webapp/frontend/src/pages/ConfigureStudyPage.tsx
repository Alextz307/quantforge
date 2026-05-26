import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { SubmitFailureAlert } from "@/components/forms/SubmitFailureAlert";
import { QueryRenderer } from "@/components/QueryRenderer";
import { YamlEditor } from "@/components/YamlEditor";
import { StudySpecFormatHelp } from "@/components/StudySpecFormatHelp";
import { SubmitJobError, useSubmitJob, type ValidationErrorItem } from "@/api/jobs";
import { useConfigDetail, useConfigList, type ConfigEntry } from "@/api/configs";
import {
  SaveUploadError,
  useDeleteStudyUpload,
  useSaveStudyUpload,
  useStudyUpload,
  useStudyUploads,
  useValidateStudySpec,
  type StudySpecUploadSummary,
} from "@/api/studyUploads";
import { STUDY_SPEC_SKELETON } from "@/lib/studySpecSkeleton";
import { jobDetailPath } from "@/lib/routes";

type SourceMode = "library" | "uploads" | "new";

const SLUG_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const SLUG_FORMAT_ERROR: ValidationErrorItem = {
  loc: ["slug"],
  msg: "Slug must start with a letter/digit and contain only letters, digits, _ and -",
  type: "value_error",
};
const VALIDATE_DEBOUNCE_MS = 500;
// Mirrors ``StudySpecUploadCreate.yaml`` Field(max_length=131072) on the
// backend so the file picker rejects oversized inputs before the request.
const MAX_YAML_BYTES = 131072;

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
  const uploadsQuery = useStudyUploads();
  const saveUpload = useSaveStudyUpload();
  const deleteUpload = useDeleteStudyUpload();
  const validateSpec = useValidateStudySpec();

  const [mode, setMode] = useState<SourceMode>("library");
  const [librarySpecName, setLibrarySpecName] = useState("");
  const [uploadSlug, setUploadSlug] = useState("");
  const [newSlug, setNewSlug] = useState("");
  const [editorYaml, setEditorYaml] = useState(STUDY_SPEC_SKELETON);
  const [validationErrors, setValidationErrors] = useState<readonly ValidationErrorItem[]>([]);

  const [forceRerun, setForceRerun] = useState(false);
  const [skipCompares, setSkipCompares] = useState(false);
  const [skipHoldoutEval, setSkipHoldoutEval] = useState(false);
  const [onlyLegsRaw, setOnlyLegsRaw] = useState("");
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const [clientErrors, setClientErrors] = useState<readonly ValidationErrorItem[]>([]);

  const libraryPreview = useConfigDetail(
    "study",
    mode === "library" && librarySpecName !== "" ? librarySpecName : null,
  );
  const uploadPreview = useStudyUpload(mode === "uploads" && uploadSlug !== "" ? uploadSlug : null);

  // Debounced server validation only fires on the editable ``new`` tab — preview
  // panes are read-only snapshots of artifacts the backend already accepted, so
  // re-validating them adds latency without informational value.
  useEffect(() => {
    if (mode !== "new") {
      setValidationErrors([]);
      return;
    }
    const handle = window.setTimeout(() => {
      validateSpec.mutate(
        { yaml: editorYaml },
        {
          onSuccess: (result) => {
            setValidationErrors(result.errors);
          },
        },
      );
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(handle);
    };
    // intentionally exclude validateSpec — its identity churns and we only
    // care about responding to text + mode changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, editorYaml]);

  const inlineErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );

  const validateClient = (): readonly ValidationErrorItem[] => {
    const errors: ValidationErrorItem[] = [];
    if (mode === "library" && librarySpecName === "") {
      errors.push({
        loc: ["study_payload", "spec_name"],
        msg: "Pick a study spec",
        type: "missing",
      });
    }
    if (mode === "uploads" && uploadSlug === "") {
      errors.push({
        loc: ["study_payload", "spec_name"],
        msg: "Pick an upload",
        type: "missing",
      });
    }
    if (mode === "new") {
      if (!SLUG_PATTERN.test(newSlug)) {
        errors.push(SLUG_FORMAT_ERROR);
      }
      if (validationErrors.length > 0) {
        errors.push({
          loc: ["yaml"],
          msg: "Fix the YAML errors before launching",
          type: "value_error",
        });
      }
    }
    return errors;
  };

  const resolvedSpecName = (): string => {
    if (mode === "library") return librarySpecName;
    if (mode === "uploads") return uploadSlug;
    return newSlug;
  };

  const launchJob = async (specName: string) => {
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
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setServerErrors([]);
    const local = validateClient();
    if (local.length > 0) {
      setClientErrors(local);
      return;
    }
    setClientErrors([]);
    try {
      if (mode === "new") {
        const detail = await saveUpload.mutateAsync({ slug: newSlug, yaml: editorYaml });
        await launchJob(detail.slug);
        return;
      }
      await launchJob(resolvedSpecName());
    } catch (err) {
      if (err instanceof SaveUploadError) {
        setServerErrors(err.fieldErrors);
        return;
      }
      if (err instanceof SubmitJobError) {
        setServerErrors(err.fieldErrors);
        return;
      }
      throw err;
    }
  };

  const onSaveOnly = async () => {
    setServerErrors([]);
    setClientErrors([]);
    if (mode !== "new") return;
    if (!SLUG_PATTERN.test(newSlug)) {
      setClientErrors([SLUG_FORMAT_ERROR]);
      return;
    }
    try {
      await saveUpload.mutateAsync({ slug: newSlug, yaml: editorYaml });
      // Stay on the page — switch the picker to the saved upload so the user
      // can verify before launching.
      setMode("uploads");
      setUploadSlug(newSlug);
    } catch (err) {
      if (err instanceof SaveUploadError) {
        setServerErrors(err.fieldErrors);
        return;
      }
      throw err;
    }
  };

  const onEditCopy = (sourceYaml: string, suggestedSlug: string) => {
    setMode("new");
    setNewSlug(`${suggestedSlug}-copy`);
    setEditorYaml(sourceYaml);
  };

  const onInsertLegSnippet = (snippet: string) => {
    setEditorYaml((prev) => `${prev.replace(/\n$/, "")}\n${snippet}`);
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const onFilePicked = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input synchronously so re-picking the same filename re-fires onChange.
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_YAML_BYTES) {
      setClientErrors([
        {
          loc: ["yaml"],
          msg: `File is ${String(file.size)} bytes — exceeds the ${String(MAX_YAML_BYTES)}-byte cap`,
          type: "value_error",
        },
      ]);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : "";
      setEditorYaml(text);
      // Pre-fill slug from the filename stem only when empty and the stem is a
      // valid slug — otherwise leave the user to type one.
      if (newSlug === "") {
        const stem = file.name.replace(/\.ya?ml$/i, "");
        if (SLUG_PATTERN.test(stem)) setNewSlug(stem);
      }
      setClientErrors([]);
    };
    reader.readAsText(file);
  };

  const onDeleteUpload = async () => {
    if (uploadSlug === "") return;
    if (!window.confirm(`Delete upload "${uploadSlug}"?`)) return;
    await deleteUpload.mutateAsync(uploadSlug);
    setUploadSlug("");
  };

  const submitDisabled =
    submit.isPending || saveUpload.isPending || (mode === "new" && validationErrors.length > 0);

  return (
    <Card className="max-w-6xl">
      <CardHeader>
        <CardTitle>Configure study</CardTitle>
        <CardDescription>
          Pick a library spec, reuse one of your uploads, or author a new spec inline. The
          orchestrator cross-products legs from{" "}
          <code className="font-mono">strategy × universe</code> and runs tune → walk-forward →
          holdout for each.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} noValidate className="space-y-6">
          <SourceModeTabs mode={mode} onChange={setMode} />

          {mode === "library" && (
            <section className="space-y-3">
              <Label htmlFor="study-spec">Library spec</Label>
              <QueryRenderer query={specsQuery} errorTitle="Failed to load library specs">
                {(specs) => (
                  <SpecPicker specs={specs} value={librarySpecName} onChange={setLibrarySpecName} />
                )}
              </QueryRenderer>
              {libraryPreview.data && (
                <ReadOnlyEditorWithActions
                  yaml={libraryPreview.data.raw}
                  onEditCopy={() => {
                    onEditCopy(libraryPreview.data.raw, librarySpecName);
                  }}
                />
              )}
            </section>
          )}

          {mode === "uploads" && (
            <section className="space-y-3">
              <Label htmlFor="study-upload">Your uploads</Label>
              <QueryRenderer query={uploadsQuery} errorTitle="Failed to load your uploads">
                {(uploads) => (
                  <UploadPicker uploads={uploads} value={uploadSlug} onChange={setUploadSlug} />
                )}
              </QueryRenderer>
              {uploadPreview.data && (
                <ReadOnlyEditorWithActions
                  yaml={uploadPreview.data.yaml}
                  onEditCopy={() => {
                    onEditCopy(uploadPreview.data.yaml, uploadSlug);
                  }}
                  onDelete={onDeleteUpload}
                  deleting={deleteUpload.isPending}
                />
              )}
            </section>
          )}

          {mode === "new" && (
            <section className="space-y-3">
              <div className="space-y-2">
                <div className="flex items-end gap-2">
                  <div className="flex-1 space-y-2">
                    <Label htmlFor="study-new-slug">Slug</Label>
                    <Input
                      id="study-new-slug"
                      value={newSlug}
                      placeholder="my_first_study"
                      onChange={(e) => {
                        setNewSlug(e.target.value);
                      }}
                    />
                  </div>
                  <input
                    ref={fileInputRef}
                    data-testid="yaml-file-input"
                    type="file"
                    accept=".yaml,.yml,application/x-yaml,text/yaml"
                    className="hidden"
                    onChange={onFilePicked}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Upload .yaml
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Becomes the spec name. Letters, digits, <code>_</code>, <code>-</code>. Cannot
                  collide with a file under <code className="font-mono">config/study/</code>.
                </p>
              </div>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
                <YamlEditor value={editorYaml} onChange={setEditorYaml} errors={validationErrors} />
                <StudySpecFormatHelp onInsertLeg={onInsertLegSnippet} />
              </div>
              <ValidationStatus pending={validateSpec.isPending} errors={validationErrors} />
            </section>
          )}

          <FlagsAndOnlyLegs
            forceRerun={forceRerun}
            skipCompares={skipCompares}
            skipHoldoutEval={skipHoldoutEval}
            onlyLegsRaw={onlyLegsRaw}
            onForceRerun={setForceRerun}
            onSkipCompares={setSkipCompares}
            onSkipHoldoutEval={setSkipHoldoutEval}
            onOnlyLegsRaw={setOnlyLegsRaw}
          />

          <ServerErrorList errors={inlineErrors} />
          <SubmitFailureAlert mutation={submit} />

          <div className="flex flex-wrap justify-end gap-2">
            {mode === "new" && (
              <Button
                type="button"
                variant="outline"
                disabled={saveUpload.isPending || validationErrors.length > 0}
                onClick={() => {
                  void onSaveOnly();
                }}
              >
                {saveUpload.isPending ? "Saving…" : "Save"}
              </Button>
            )}
            <Button type="submit" disabled={submitDisabled}>
              {submit.isPending || saveUpload.isPending
                ? "Launching…"
                : mode === "new"
                  ? "Save & launch"
                  : "Launch study"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

interface SourceModeTabsProps {
  mode: SourceMode;
  onChange: (next: SourceMode) => void;
}

function SourceModeTabs({ mode, onChange }: SourceModeTabsProps) {
  const tabs: { id: SourceMode; label: string }[] = [
    { id: "library", label: "Library" },
    { id: "uploads", label: "My uploads" },
    { id: "new", label: "New spec" },
  ];
  return (
    <div className="flex gap-1 border-b border-input pb-2" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          type="button"
          aria-selected={mode === t.id}
          onClick={() => {
            onChange(t.id);
          }}
          className={`rounded-md px-3 py-1.5 text-sm transition ${
            mode === t.id
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-muted"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
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
        No study specs found under <code className="font-mono">config/study/</code>. Switch to{" "}
        <em>New spec</em> to author one.
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

interface UploadPickerProps {
  uploads: readonly StudySpecUploadSummary[];
  value: string;
  onChange: (slug: string) => void;
}

function UploadPicker({ uploads, value, onChange }: UploadPickerProps) {
  if (uploads.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        You haven't saved any uploads yet. Switch to <em>New spec</em> to author one.
      </p>
    );
  }
  return (
    <select
      id="study-upload"
      className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
      value={value}
      onChange={(e) => {
        onChange(e.target.value);
      }}
    >
      <option value="">— pick an upload —</option>
      {uploads.map((u) => (
        <option key={u.slug} value={u.slug}>
          {u.slug}
        </option>
      ))}
    </select>
  );
}

interface ReadOnlyEditorWithActionsProps {
  yaml: string;
  onEditCopy: () => void;
  onDelete?: () => void;
  deleting?: boolean;
}

function ReadOnlyEditorWithActions({
  yaml,
  onEditCopy,
  onDelete,
  deleting = false,
}: ReadOnlyEditorWithActionsProps) {
  return (
    <div className="space-y-2">
      <YamlEditor value={yaml} onChange={() => undefined} readOnly height="320px" />
      <div className="flex justify-end gap-2">
        {onDelete && (
          <Button type="button" variant="outline" size="sm" onClick={onDelete} disabled={deleting}>
            {deleting ? "Deleting…" : "Delete"}
          </Button>
        )}
        <Button type="button" variant="outline" size="sm" onClick={onEditCopy}>
          Edit a copy
        </Button>
      </div>
    </div>
  );
}

interface ValidationStatusProps {
  pending: boolean;
  errors: readonly ValidationErrorItem[];
}

function ValidationStatus({ pending, errors }: ValidationStatusProps) {
  if (pending) {
    return <p className="text-xs text-muted-foreground">Validating…</p>;
  }
  if (errors.length === 0) {
    return <p className="text-xs text-emerald-700">✓ No validation errors.</p>;
  }
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-destructive">
        {errors.length} validation error{errors.length === 1 ? "" : "s"}
      </summary>
      <ul className="mt-1 space-y-0.5 pl-4">
        {errors.map((err, idx) => (
          <li key={idx} className="font-mono">
            <strong>{err.loc.join(".")}</strong>: {err.msg}
          </li>
        ))}
      </ul>
    </details>
  );
}

interface FlagsAndOnlyLegsProps {
  forceRerun: boolean;
  skipCompares: boolean;
  skipHoldoutEval: boolean;
  onlyLegsRaw: string;
  onForceRerun: (next: boolean) => void;
  onSkipCompares: (next: boolean) => void;
  onSkipHoldoutEval: (next: boolean) => void;
  onOnlyLegsRaw: (next: string) => void;
}

function FlagsAndOnlyLegs({
  forceRerun,
  skipCompares,
  skipHoldoutEval,
  onlyLegsRaw,
  onForceRerun,
  onSkipCompares,
  onSkipHoldoutEval,
  onOnlyLegsRaw,
}: FlagsAndOnlyLegsProps) {
  return (
    <div className="space-y-4 border-t border-input pt-4">
      <div className="flex flex-col gap-2">
        <Label htmlFor="study-only-legs">Only legs (optional)</Label>
        <Input
          id="study-only-legs"
          value={onlyLegsRaw}
          placeholder="e.g. AdaptiveBollinger__spy_daily_5y, MomentumGatekeeper__qqq_daily_5y"
          onChange={(e) => {
            onOnlyLegsRaw(e.target.value);
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
                onForceRerun(e.target.checked);
              }}
            />
            Force rerun — ignore <code className="font-mono">is_complete</code> markers and re-run
            every leg from scratch
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={skipCompares}
              onChange={(e) => {
                onSkipCompares(e.target.checked);
              }}
            />
            Skip per-universe cross-strategy compares
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={skipHoldoutEval}
              onChange={(e) => {
                onSkipHoldoutEval(e.target.checked);
              }}
            />
            Skip holdout-eval on every leg
          </label>
        </div>
      </div>
    </div>
  );
}
