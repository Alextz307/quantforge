import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ServerErrorList } from "@/components/forms/ServerErrorList";
import { QueryRenderer } from "@/components/QueryRenderer";
import { YamlEditor } from "@/components/YamlEditor";
import { type ValidationErrorItem } from "@/api/jobs";
import {
  SaveUploadError,
  useDeleteUniverseUpload,
  useSaveUniverseUpload,
  useUniverseSpecSchema,
  useUniverseUpload,
  useUniverseUploads,
  useValidateUniverseSpec,
  type UniverseSpecUploadSummary,
} from "@/api/universeUploads";
import { UNIVERSE_SPEC_SKELETON } from "@/lib/universeSpecSkeleton";

type SourceMode = "uploads" | "new";

const SLUG_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const SLUG_FORMAT_ERROR: ValidationErrorItem = {
  loc: ["slug"],
  msg: "Slug must start with a letter/digit and contain only letters, digits, _ and -",
  type: "value_error",
};
const VALIDATE_DEBOUNCE_MS = 500;
// Mirrors ``UniverseSpecUploadCreate.yaml`` Field(max_length=131072) so
// the file picker rejects oversized inputs client-side.
const MAX_YAML_BYTES = 131072;

export function ConfigureUniversePage() {
  const uploadsQuery = useUniverseUploads();
  const saveUpload = useSaveUniverseUpload();
  const deleteUpload = useDeleteUniverseUpload();
  const validateSpec = useValidateUniverseSpec();
  // Pre-warm the schema cache so YamlEditor autocomplete is instant on first edit.
  useUniverseSpecSchema();

  const [mode, setMode] = useState<SourceMode>("uploads");
  const [uploadSlug, setUploadSlug] = useState("");
  const [newSlug, setNewSlug] = useState("");
  const [editorYaml, setEditorYaml] = useState(UNIVERSE_SPEC_SKELETON);
  const [validationErrors, setValidationErrors] = useState<readonly ValidationErrorItem[]>([]);
  const [serverErrors, setServerErrors] = useState<readonly ValidationErrorItem[]>([]);
  const [clientErrors, setClientErrors] = useState<readonly ValidationErrorItem[]>([]);

  const uploadPreview = useUniverseUpload(
    mode === "uploads" && uploadSlug !== "" ? uploadSlug : null,
  );

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
    // intentionally exclude validateSpec - its identity churns and we only
    // care about responding to text + mode changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, editorYaml]);

  const inlineErrors = useMemo(
    () => [...clientErrors, ...serverErrors],
    [clientErrors, serverErrors],
  );

  const onSave = async (e: FormEvent) => {
    e.preventDefault();
    setServerErrors([]);
    if (mode !== "new") return;
    if (!SLUG_PATTERN.test(newSlug)) {
      setClientErrors([SLUG_FORMAT_ERROR]);
      return;
    }
    if (validationErrors.length > 0) {
      setClientErrors([
        { loc: ["yaml"], msg: "Fix the YAML errors before saving", type: "value_error" },
      ]);
      return;
    }
    setClientErrors([]);
    try {
      const detail = await saveUpload.mutateAsync({ slug: newSlug, yaml: editorYaml });
      // Switch the picker to the new upload so the user can verify it landed.
      setMode("uploads");
      setUploadSlug(detail.slug);
    } catch (err: unknown) {
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

  const fileInputRef = useRef<HTMLInputElement>(null);

  const onFilePicked = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_YAML_BYTES) {
      setClientErrors([
        {
          loc: ["yaml"],
          msg: `File is ${String(file.size)} bytes - exceeds the ${String(MAX_YAML_BYTES)}-byte cap`,
          type: "value_error",
        },
      ]);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : "";
      setEditorYaml(text);
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

  const saveDisabled = saveUpload.isPending || (mode === "new" && validationErrors.length > 0);

  return (
    <Card className="max-w-6xl">
      <CardHeader>
        <CardTitle>Configure universe</CardTitle>
        <CardDescription>
          Manage reusable universe specs. A universe pins{" "}
          <code className="font-mono">data.source x tickers x interval x date window</code> so study
          legs can reference it by slug.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSave} noValidate className="space-y-6">
          <SourceModeTabs mode={mode} onChange={setMode} />

          {mode === "uploads" && (
            <section className="space-y-3">
              <Label htmlFor="universe-upload">Your uploads</Label>
              <QueryRenderer query={uploadsQuery} errorTitle="Failed to load your uploads">
                {(uploads) => (
                  <UploadPicker uploads={uploads} value={uploadSlug} onChange={setUploadSlug} />
                )}
              </QueryRenderer>
              {uploadPreview.data && (
                <div className="space-y-2">
                  <div className="rounded-md border bg-muted/30">
                    <YamlEditor
                      value={uploadPreview.data.yaml}
                      onChange={() => {
                        /* read-only */
                      }}
                      readOnly
                      height="280px"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        onEditCopy(uploadPreview.data.yaml, uploadSlug);
                      }}
                    >
                      Edit as new
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={deleteUpload.isPending}
                      onClick={onDeleteUpload}
                      data-testid="universe-upload-delete"
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              )}
            </section>
          )}

          {mode === "new" && (
            <section className="space-y-3">
              <div className="flex items-end gap-2">
                <div className="flex-1 space-y-2">
                  <Label htmlFor="universe-new-slug">Slug</Label>
                  <Input
                    id="universe-new-slug"
                    value={newSlug}
                    placeholder="my_universe"
                    onChange={(e) => {
                      setNewSlug(e.target.value);
                    }}
                  />
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".yaml,.yml,application/yaml,text/yaml"
                  hidden
                  onChange={onFilePicked}
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Upload file
                </Button>
              </div>
              <div className="rounded-md border">
                <YamlEditor
                  value={editorYaml}
                  onChange={(next) => {
                    setEditorYaml(next);
                  }}
                  height="360px"
                />
              </div>
              {validationErrors.length > 0 && <ServerErrorList errors={validationErrors} />}
            </section>
          )}

          {inlineErrors.length > 0 && <ServerErrorList errors={inlineErrors} />}

          {mode === "new" && (
            <div className="flex justify-end gap-2">
              <Button type="submit" disabled={saveDisabled} data-testid="universe-save">
                {saveUpload.isPending ? "Saving..." : "Save upload"}
              </Button>
            </div>
          )}
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
  const tabs: { value: SourceMode; label: string }[] = [
    { value: "uploads", label: "Your uploads" },
    { value: "new", label: "New spec" },
  ];
  return (
    <div className="flex gap-2 border-b" role="tablist">
      {tabs.map((tab) => {
        const active = tab.value === mode;
        return (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => {
              onChange(tab.value);
            }}
            className={`px-3 py-2 text-sm ${
              active ? "border-b-2 border-primary font-medium" : "text-muted-foreground"
            }`}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

interface UploadPickerProps {
  uploads: readonly UniverseSpecUploadSummary[];
  value: string;
  onChange: (next: string) => void;
}

function UploadPicker({ uploads, value, onChange }: UploadPickerProps) {
  if (uploads.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        You haven&apos;t uploaded any universe specs yet. Switch to <em>New spec</em> to author one.
      </p>
    );
  }
  return (
    <select
      id="universe-upload"
      className="w-full rounded-md border px-3 py-2 text-sm"
      value={value}
      onChange={(e) => {
        onChange(e.target.value);
      }}
      data-testid="universe-upload-picker"
    >
      <option value="">Select an upload...</option>
      {uploads.map((u) => (
        <option key={u.slug} value={u.slug}>
          {u.slug}
        </option>
      ))}
    </select>
  );
}
