import { FilterField } from "@/components/FilterField";
import { ALL_OPTION } from "@/lib/filters";

export interface FilterSelectProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  allLabel: string;
  options: readonly string[];
  optionLabel?: (value: string) => string;
}

export function FilterSelect({
  id,
  label,
  value,
  onChange,
  allLabel,
  options,
  optionLabel,
}: FilterSelectProps) {
  return (
    <FilterField id={id} label={label}>
      <select
        id={id}
        className="h-9 rounded-md border bg-background px-2 text-sm"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
        }}
      >
        <option value={ALL_OPTION}>{allLabel}</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {optionLabel ? optionLabel(o) : o}
          </option>
        ))}
      </select>
    </FilterField>
  );
}
