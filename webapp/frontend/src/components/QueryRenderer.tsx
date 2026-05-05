import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export interface QueryRendererProps<T> {
  query: UseQueryResult<T>;
  errorTitle: string;
  loadingMessage?: string;
  children: (data: T) => ReactNode;
}

export function QueryRenderer<T>({
  query,
  errorTitle,
  loadingMessage = "Loading…",
  children,
}: QueryRendererProps<T>) {
  if (query.isPending) {
    return <p className="text-sm text-muted-foreground">{loadingMessage}</p>;
  }
  if (query.isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>{errorTitle}</AlertTitle>
        <AlertDescription>{query.error.message}</AlertDescription>
      </Alert>
    );
  }
  return <>{children(query.data)}</>;
}
