import { useId, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/cn";
import { parseStringList } from "@/lib/schemas/configureForm";
import type { StrategyParam, StrategySchema } from "@/api/strategies";

interface StrategyParamsEditorProps {
  schema: StrategySchema;
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
  errorsByLoc?: ReadonlyMap<string, string> | undefined;
  disabled?: boolean | undefined;
}

/**
 * Per-strategy params editor. Renders a typed input per simple ``ParamKind``
 * (int/float/str/bool/enum) and a JSON-editor textarea for ``complex``
 * params (lists, Optional, Path, etc.). Server-side validation via
 * /api/configs/validate is the source of truth — this component only
 * sanitises into a shape Pydantic can parse.
 */
export function StrategyParamsEditor({
  schema,
  values,
  onChange,
  errorsByLoc,
  disabled,
}: StrategyParamsEditorProps) {
  if (schema.params.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        This strategy takes no constructor parameters.
      </p>
    );
  }

  const setValue = (name: string, value: unknown) => {
    onChange({ ...values, [name]: value });
  };

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {schema.params.map((param) => (
        <ParamField
          key={param.name}
          param={param}
          value={values[param.name]}
          onChange={(v) => {
            setValue(param.name, v);
          }}
          errorMsg={errorsByLoc?.get(`strategy.params.${param.name}`)}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

interface ParamFieldProps {
  param: StrategyParam;
  value: unknown;
  onChange: (value: unknown) => void;
  errorMsg?: string | undefined;
  disabled?: boolean | undefined;
}

function ParamField({ param, value, onChange, errorMsg, disabled }: ParamFieldProps) {
  const id = useId();
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="flex items-center gap-2">
        <span className="font-mono text-xs">{param.name}</span>
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
          {param.kind}
          {param.required ? " · required" : ""}
        </span>
      </Label>
      <ParamInput id={id} param={param} value={value} onChange={onChange} disabled={disabled} />
      {errorMsg && <p className="text-xs text-rose-600">{errorMsg}</p>}
    </div>
  );
}

interface ParamInputProps {
  id: string;
  param: StrategyParam;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean | undefined;
}

function ParamInput({ id, param, value, onChange, disabled }: ParamInputProps) {
  const placeholder = describeDefault(param);
  switch (param.kind) {
    case "int":
      return (
        <Input
          id={id}
          type="number"
          step={1}
          disabled={disabled}
          placeholder={placeholder}
          value={asScalarInputValue(value)}
          onChange={(e) => {
            onChange(e.target.value === "" ? undefined : Number.parseInt(e.target.value, 10));
          }}
        />
      );
    case "float":
      return (
        <Input
          id={id}
          type="number"
          step="any"
          disabled={disabled}
          placeholder={placeholder}
          value={asScalarInputValue(value)}
          onChange={(e) => {
            onChange(e.target.value === "" ? undefined : Number.parseFloat(e.target.value));
          }}
        />
      );
    case "str":
      return (
        <Input
          id={id}
          type="text"
          disabled={disabled}
          placeholder={placeholder}
          value={typeof value === "string" ? value : ""}
          onChange={(e) => {
            onChange(e.target.value === "" ? undefined : e.target.value);
          }}
        />
      );
    case "bool":
      return (
        <input
          id={id}
          type="checkbox"
          disabled={disabled}
          className="h-5 w-5"
          checked={value === true}
          onChange={(e) => {
            onChange(e.target.checked);
          }}
        />
      );
    case "str_list":
      return (
        <Input
          id={id}
          type="text"
          disabled={disabled}
          placeholder={strListPlaceholder(param.name)}
          value={
            Array.isArray(value) && value.every((v) => typeof v === "string")
              ? value.join(", ")
              : ""
          }
          onChange={(e) => {
            const items = parseStringList(e.target.value);
            onChange(items.length === 0 ? undefined : items);
          }}
        />
      );
    case "enum":
      return (
        <select
          id={id}
          disabled={disabled}
          required={param.required}
          className={cn(
            "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
          )}
          value={typeof value === "string" ? value : ""}
          onChange={(e) => {
            onChange(e.target.value === "" ? undefined : e.target.value);
          }}
        >
          <option value="">{emptyOptionLabel(param)}</option>
          {(param.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      );
    case "complex":
      return (
        <JsonInput
          id={id}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
        />
      );
  }
}

interface JsonInputProps {
  id: string;
  value: unknown;
  onChange: (value: unknown) => void;
  placeholder: string;
  disabled?: boolean | undefined;
}

// Holds the user's literal keystrokes in local state so partial JSON
// ("[", '["foo') survives re-renders; commits to the parent only on a
// successful JSON.parse, leaving the last-good value otherwise.
function JsonInput({ id, value, onChange, placeholder, disabled }: JsonInputProps) {
  const [text, setText] = useState(() => (value === undefined ? "" : JSON.stringify(value)));
  const [parseError, setParseError] = useState<string | null>(null);
  return (
    <div className="space-y-1">
      <textarea
        id={id}
        disabled={disabled}
        rows={3}
        className="font-mono w-full rounded-md border border-input bg-background px-3 py-2 text-xs"
        placeholder={placeholder}
        value={text}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          if (next.trim() === "") {
            setParseError(null);
            onChange(undefined);
            return;
          }
          try {
            onChange(JSON.parse(next));
            setParseError(null);
          } catch (err) {
            setParseError(err instanceof Error ? err.message : "invalid JSON");
          }
        }}
      />
      {parseError && <p className="text-xs text-amber-600">JSON: {parseError}</p>}
    </div>
  );
}

function asScalarInputValue(value: unknown): number | string {
  if (typeof value === "number") return value;
  if (typeof value === "string") return value;
  return "";
}

function describeDefault(param: StrategyParam): string {
  const def = param.default;
  if (def === null || def === undefined) {
    return param.required ? "required" : "default: —";
  }
  return `default: ${formatPrimitive(def)}`;
}

function emptyOptionLabel(param: StrategyParam): string {
  if (param.required) return "— select —";
  if (param.nullable) return "— none —";
  return "— use default —";
}

function strListPlaceholder(paramName: string): string {
  if (paramName === "tickers" || paramName.endsWith("_tickers")) {
    return "comma- or space-separated tickers, e.g. QQQ, TLT, GLD";
  }
  if (paramName.endsWith("_columns")) {
    return "comma- or space-separated, e.g. rsi_14, macd_signal, ma_ratio";
  }
  return "comma- or space-separated";
}

function formatPrimitive(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}
