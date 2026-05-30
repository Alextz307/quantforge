import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { JobRow } from "@/api/jobs";
import { jobArtifactLink } from "@/lib/jobArtifact";
import { cn } from "@/lib/cn";

interface JobArtifactLinkProps {
  job: JobRow;
  compact?: boolean;
  className?: string | undefined;
  fallback?: ReactNode;
}

export function JobArtifactLink({
  job,
  compact = false,
  className,
  fallback = null,
}: JobArtifactLinkProps) {
  const link = jobArtifactLink(job);
  if (!link) return <>{fallback}</>;
  return (
    <Link
      to={link.to}
      className={cn("text-primary hover:underline", className)}
      data-testid="job-view-run-link"
    >
      {compact ? "view ->" : link.label}
    </Link>
  );
}
