import { useId } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/cn";
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
    case "enum":
      return (
        <select
          id={id}
          disabled={disabled}
          className={cn(
            "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
          )}
          value={typeof value === "string" ? value : ""}
          onChange={(e) => {
            onChange(e.target.value === "" ? undefined : e.target.value);
          }}
        >
          <option value="">— use default —</option>
          {(param.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      );
    case "complex":
      return (
        <textarea
          id={id}
          disabled={disabled}
          rows={3}
          className="font-mono w-full rounded-md border border-input bg-background px-3 py-2 text-xs"
          placeholder={placeholder}
          value={value === undefined ? "" : JSON.stringify(value)}
          onChange={(e) => {
            const text = e.target.value.trim();
            if (text === "") {
              onChange(undefined);
              return;
            }
            try {
              onChange(JSON.parse(text));
            } catch {
              // Keep the previous value; the next valid JSON parse will
              // commit. Server-side validate gives the user a clear error
              // if they submit broken JSON before fixing it.
            }
          }}
        />
      );
  }
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

function formatPrimitive(value: unknown): string {
  // Backend already serialises complex defaults via repr() — they arrive
  // as plain strings here. This helper just guards against object slips.
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}
