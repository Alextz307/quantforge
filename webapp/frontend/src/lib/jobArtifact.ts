import type { JobRow } from "@/api/jobs";
import {
  comparisonDetailPath,
  holdoutDetailPath,
  hpoDetailPath,
  runDetailPath,
  studyDetailPath,
} from "@/lib/routes";

export interface JobArtifactLink {
  to: string;
  label: string;
}

export function jobArtifactLink(job: JobRow): JobArtifactLink | null {
  if (job.status !== "completed" || job.experiment_id === null) return null;
  switch (job.kind) {
    case "tune":
      return { to: hpoDetailPath(job.experiment_id), label: "View study ->" };
    case "compare":
      return { to: comparisonDetailPath(job.experiment_id), label: "View comparison ->" };
    case "holdout":
      return { to: holdoutDetailPath(job.experiment_id), label: "View holdout ->" };
    case "run":
      return { to: runDetailPath(job.experiment_id), label: "View run ->" };
    case "study":
      return { to: studyDetailPath(job.experiment_id), label: "View study ->" };
  }
}
