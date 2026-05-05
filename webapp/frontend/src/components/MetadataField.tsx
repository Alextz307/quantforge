import type { ReactNode } from "react";

export interface MetadataFieldProps {
  label: string;
  value: ReactNode;
}

export function MetadataField({ label, value }: MetadataFieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="text-sm font-mono break-all">{value}</span>
    </div>
  );
}
