import type { ReactNode } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { SubmitJobError } from "@/api/jobs";

interface SubmitFailureAlertProps {
  mutation: { isError: boolean; error: Error | null };
}

/**
 * Renders the generic submit-failure Alert iff the mutation errored AND the
 * error is NOT a SubmitJobError — those carry structured field errors that
 * <ServerErrorList> already surfaces. This guard prevents double-rendering.
 */
export function SubmitFailureAlert({ mutation }: SubmitFailureAlertProps): ReactNode {
  if (!mutation.isError) return null;
  if (mutation.error instanceof SubmitJobError) return null;
  const message = mutation.error?.message ?? "Failed to submit job";
  return (
    <Alert variant="destructive">
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
}
