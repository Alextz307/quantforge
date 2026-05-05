import { FilterField } from "@/components/FilterField";
import { Input } from "@/components/ui/input";

export interface FilterDateProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
}

export function FilterDate({ id, label, value, onChange }: FilterDateProps) {
  return (
    <FilterField id={id} label={label}>
      <Input
        id={id}
        type="date"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
        }}
      />
    </FilterField>
  );
}
