import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { ValidationErrorItem } from "@/api/jobs";

interface ServerErrorListProps {
  errors: readonly ValidationErrorItem[];
}

export function ServerErrorList({ errors }: ServerErrorListProps): ReactNode {
  if (errors.length === 0) return null;
  return (
    <Alert variant="destructive">
      <AlertTitle>Backend rejected the config</AlertTitle>
      <AlertDescription>
        <ul className="list-disc space-y-1 pl-4">
          {errors.map((err, idx) => (
            <li key={idx} className="font-mono text-xs">
              <strong>{err.loc.join(".")}</strong>: {err.msg}
            </li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}
