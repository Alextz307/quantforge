import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/cn";
import { useConfigDetail, useConfigList } from "@/api/configs";
import { formatDate } from "@/lib/format";
import type { IntervalValue } from "@/lib/schemas/configureForm";
import { isIntervalValue } from "@/lib/schemas/configureForm";

export interface UniversePreset {
  tickers: string;
  start: string;
  end: string;
  interval: IntervalValue;
}

interface UniversePickerProps {
  onApply: (preset: UniversePreset) => void;
  disabled?: boolean | undefined;
  className?: string | undefined;
}

/**
 * Dropdown over ``config/universes/*.yaml``. Selecting a name and clicking
 * "Apply" loads the universe YAML and writes its data block back into
 * the form's tickers/start/end/interval fields. The user can edit any of
 * those afterward - universe is a *preset*, not a binding.
 */
export function UniversePicker({ onApply, disabled, className }: UniversePickerProps) {
  const list = useConfigList("universe");
  const [name, setName] = useState<string | null>(null);
  const detail = useConfigDetail("universe", name);
  const isLoading = detail.isFetching;

  const handleApply = () => {
    const parsed = detail.data?.parsed;
    if (!parsed) return;
    const preset = parsedToPreset(parsed);
    if (preset) onApply(preset);
  };

  return (
    <div className={cn("space-y-2", className)}>
      <Label htmlFor="universe-picker" className="text-xs uppercase tracking-wide">
        Universe preset
      </Label>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <select
          id="universe-picker"
          disabled={disabled || list.isLoading}
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm sm:max-w-xs"
          value={name ?? ""}
          onChange={(e) => {
            setName(e.target.value || null);
          }}
        >
          <option value="">- select to preview -</option>
          {list.data?.map((entry) => (
            <option key={entry.name} value={entry.name}>
              {entry.name}
            </option>
          ))}
        </select>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={disabled || isLoading || !detail.data?.parsed}
          onClick={handleApply}
        >
          {isLoading ? "Loading..." : "Apply preset"}
        </Button>
      </div>
      {detail.data?.parse_error && (
        <p className="text-xs text-rose-600 dark:text-rose-400">
          YAML parse error: {detail.data.parse_error}
        </p>
      )}
    </div>
  );
}

function parsedToPreset(parsed: Readonly<Record<string, unknown>>): UniversePreset | null {
  const data = parsed.data;
  if (!data || typeof data !== "object") return null;
  const d = data as Record<string, unknown>;
  const tickers = Array.isArray(d.tickers) ? d.tickers.filter((t) => typeof t === "string") : [];
  // Universe YAMLs may carry full timestamps (e.g. "2020-01-01T00:00:00")
  // but the form's date input wants YYYY-MM-DD only.
  const start = typeof d.start === "string" ? formatDate(d.start) : "";
  const end = typeof d.end === "string" ? formatDate(d.end) : "";
  const intervalRaw = typeof d.interval === "string" ? d.interval : "daily";
  const interval: IntervalValue = isIntervalValue(intervalRaw) ? intervalRaw : "daily";
  if (tickers.length === 0 || !start || !end) return null;
  return { tickers: tickers.join(", "), start, end, interval };
}
