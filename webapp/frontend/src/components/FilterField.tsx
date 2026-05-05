import type { ReactNode } from "react";
import { Label } from "@/components/ui/label";

export interface FilterFieldProps {
  id: string;
  label: string;
  children: ReactNode;
}

export function FilterField({ id, label, children }: FilterFieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor={id}>{label}</Label>
      {children}
    </div>
  );
}
