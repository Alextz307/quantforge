import { useEffect, useRef } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import type * as MonacoNs from "monaco-editor";
import type { editor } from "monaco-editor";
import type { ValidationErrorItem } from "@/api/configs";
import { useTheme } from "@/lib/theme";
import type { ResolvedTheme } from "@/lib/themeStorage";

const MONACO_THEME: Record<ResolvedTheme, "vs" | "vs-dark"> = {
  light: "vs",
  dark: "vs-dark",
};

// ``@monaco-editor/react`` exports ``Monaco`` as ``typeof
// monaco-editor/esm/vs/editor/editor.api`` — a sub-path import TypeScript
// (with the project's ``moduleResolution: bundler``) cannot resolve, so the
// alias falls through to ``any`` and trips strict-no-unsafe lint rules. We
// bypass that by re-typing the captured instance as the public ``monaco-editor``
// module, which has the same runtime shape.
type MonacoApi = typeof MonacoNs;

interface YamlEditorProps {
  value: string;
  onChange: (next: string) => void;
  readOnly?: boolean;
  errors?: readonly ValidationErrorItem[];
  height?: string;
}

/**
 * Monaco-backed YAML editor with server-driven inline error markers.
 *
 * We deliberately bypass ``monaco-yaml`` and its WebWorker pipeline: the
 * authoritative validator lives in the backend (``StudySpec`` + path-existence
 * + registered-strategy checks), so we use Monaco purely for editing and
 * project server errors back as native ``IMarkerData`` decorations. That
 * keeps the frontend bundle slimmer (no yaml.worker) and the error feed
 * lockstep with what the job submission would actually see.
 *
 * Each ``ValidationErrorItem`` with a numeric line hint becomes a red
 * squiggle; errors whose ``loc`` we can't pin to a line surface in the page's
 * ``<ServerErrorList>`` instead.
 */
export function YamlEditor({
  value,
  onChange,
  readOnly = false,
  errors = [],
  height = "440px",
}: YamlEditorProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<MonacoApi | null>(null);
  const { resolvedTheme } = useTheme();
  const monacoTheme = MONACO_THEME[resolvedTheme];

  const onMount: OnMount = (editorInstance, monaco) => {
    editorRef.current = editorInstance;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    monacoRef.current = monaco;
  };

  useEffect(() => {
    const monaco = monacoRef.current;
    if (!monaco) return;
    monaco.editor.setTheme(monacoTheme);
  }, [monacoTheme]);

  useEffect(() => {
    const editorInstance = editorRef.current;
    const monaco = monacoRef.current;
    if (!editorInstance || !monaco) return;
    const model = editorInstance.getModel();
    if (!model) return;
    const markers: editor.IMarkerData[] = errors
      .filter((err) => locToLine(err.loc, value) !== null)
      .map((err) => {
        const line = locToLine(err.loc, value) ?? 1;
        return {
          severity: monaco.MarkerSeverity.Error,
          message: `${err.loc.join(".")}: ${err.msg}`,
          startLineNumber: line,
          startColumn: 1,
          endLineNumber: line,
          endColumn: model.getLineMaxColumn(line),
        };
      });
    monaco.editor.setModelMarkers(model, "server", markers);
  }, [errors, value]);

  return (
    <div className="overflow-hidden rounded-md border border-input">
      <Editor
        height={height}
        defaultLanguage="yaml"
        value={value}
        onChange={(next) => {
          onChange(next ?? "");
        }}
        onMount={onMount}
        options={{
          readOnly,
          fontSize: 13,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          tabSize: 2,
          insertSpaces: true,
          wordWrap: "on",
          renderWhitespace: "selection",
        }}
        theme={monacoTheme}
      />
    </div>
  );
}

/**
 * Best-effort mapping from a Pydantic-shaped ``loc`` path to a 1-indexed
 * Monaco line number.
 *
 * Top-level keys (``name``, ``output_dir``) are located by a flush
 * ``^<key>:`` line match. ``legs[i].*`` errors are pinned to the ``i``-th
 * ``- strategy:`` line so the squiggle lands on something the reader can see
 * even if the precise sub-field can't be uniquely identified.
 *
 * Errors that can't be located fall through to ``null`` — the calling page
 * surfaces them in the structured error list under the editor instead.
 */
function locToLine(loc: readonly string[], text: string): number | null {
  const head = loc[0];
  if (head === undefined || head === "yaml") return 1;
  const lines = text.split("\n");
  if (loc.length === 1) {
    const idx = lines.findIndex((l) => new RegExp(`^${head}\\s*:`).test(l));
    return idx === -1 ? null : idx + 1;
  }
  const indexPart = loc[1];
  if (head === "legs" && indexPart !== undefined && /^\d+$/.test(indexPart)) {
    const legIndex = Number(indexPart);
    let seen = -1;
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i] ?? "";
      if (/^\s*-\s*strategy\s*:/.test(line)) {
        seen += 1;
        if (seen === legIndex) return i + 1;
      }
    }
  }
  return null;
}
