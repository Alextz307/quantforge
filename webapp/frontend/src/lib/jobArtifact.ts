import type { JobRow } from "@/api/jobs";
import { hpoDetailPath, runDetailPath } from "@/lib/routes";

export interface JobArtifactLink {
  to: string;
  label: string;
}

export function jobArtifactLink(job: JobRow): JobArtifactLink | null {
  if (job.status !== "completed" || job.experiment_id === null) return null;
  if (job.kind === "tune") {
    return { to: hpoDetailPath(job.experiment_id), label: "View study →" };
  }
  return { to: runDetailPath(job.experiment_id), label: "View run →" };
}
