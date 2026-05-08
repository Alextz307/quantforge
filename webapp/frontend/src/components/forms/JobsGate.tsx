import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { usePublicSettings } from "@/api/settings";

interface JobsGateProps {
  children: ReactNode;
}

export function JobsGate({ children }: JobsGateProps): ReactNode {
  const settings = usePublicSettings();
  if (settings.isLoading) {
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  }
  if (settings.data && !settings.data.jobs_enabled) {
    return (
      <Alert variant="destructive" className="max-w-2xl">
        <AlertTitle>Job execution is disabled</AlertTitle>
        <AlertDescription>
          Set <code className="font-mono">WEBAPP_JOBS_ENABLED=true</code> and restart the backend to
          enable launching new jobs from the UI.
        </AlertDescription>
      </Alert>
    );
  }
  return <>{children}</>;
}
