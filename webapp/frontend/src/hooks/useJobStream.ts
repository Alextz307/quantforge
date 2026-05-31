import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { JobRow, JobStatus } from "@/api/jobs";
import { isTerminalStatus, jobLogDownloadUrl, jobStreamUrl } from "@/api/jobs";
import { queryKeys } from "@/api/queryKeys";
import { useEventStream, type ConnectionState } from "@/hooks/useEventStream";

export interface JobStreamSnapshot {
  logs: readonly string[];
  connection: ConnectionState;
}

interface LogFrame {
  type: "log";
  line: string;
}

interface StatusFrame {
  type: "status";
  status: JobStatus;
  exit_code?: number | null;
  experiment_id?: string | null;
}

type StreamFrame = LogFrame | StatusFrame;

/**
 * Subscribes to /api/jobs/{id}/stream and patches the per-job + jobs-list
 * caches:
 *   - patches the per-job cache via setQueryData on every status frame;
 *   - invalidates the jobs list only on terminal status (the running ``useJobs``
 *     poll keeps the list fresh while jobs are open);
 *   - on a terminal-completed status with an experiment_id, invalidates the
 *     runs list so the freshly-finished run appears in /runs.
 */
export function useJobStream(jobId: string, initialStatus: JobStatus): JobStreamSnapshot {
  const qc = useQueryClient();
  const enabled = !isTerminalStatus(initialStatus);
  const [logs, setLogs] = useState<string[]>([]);

  const handleStatus = useCallback(
    (frame: StatusFrame) => {
      qc.setQueryData<JobRow>(queryKeys.job(jobId), (prev) => {
        if (!prev) return prev;
        const nextExitCode = frame.exit_code ?? prev.exit_code;
        const nextExperimentId = frame.experiment_id ?? prev.experiment_id;
        if (
          prev.status === frame.status &&
          prev.exit_code === nextExitCode &&
          prev.experiment_id === nextExperimentId
        ) {
          return prev;
        }
        return {
          ...prev,
          status: frame.status,
          exit_code: nextExitCode,
          experiment_id: nextExperimentId,
        };
      });
      if (isTerminalStatus(frame.status)) {
        void qc.invalidateQueries({ queryKey: queryKeys.jobsAll });
        if (frame.status === "completed" && frame.experiment_id) {
          // Refresh the runs LIST only, not an open run detail's own queries
          // (detail / folds / importance): those didn't change, and refetching
          // them reflows that page's charts (e.g. disturbs the equity curve).
          void qc.invalidateQueries({
            predicate: (q) => q.queryKey[0] === "runs" && q.queryKey[1] === "page",
          });
        }
      }
    },
    [jobId, qc],
  );

  const { connection } = useEventStream<StreamFrame>({
    url: jobStreamUrl(jobId),
    parseFrame,
    enabled,
    onFrame: (frame) => {
      if (frame.type === "log") setLogs((prev) => [...prev, frame.line]);
      else handleStatus(frame);
    },
    shouldClose: (frame) => frame.type === "status" && isTerminalStatus(frame.status),
    backfillUrl: jobLogDownloadUrl(jobId),
    backfillParse: parseLogBackfill,
  });

  return { logs, connection };
}

function parseFrame(raw: string): StreamFrame | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;
  if (obj.type === "log" && typeof obj.line === "string") {
    return { type: "log", line: obj.line };
  }
  if (obj.type === "status" && typeof obj.status === "string") {
    return {
      type: "status",
      status: obj.status as JobStatus,
      exit_code: typeof obj.exit_code === "number" ? obj.exit_code : null,
      experiment_id: typeof obj.experiment_id === "string" ? obj.experiment_id : null,
    };
  }
  return null;
}

function parseLogBackfill(text: string): readonly LogFrame[] {
  const lines = text.split("\n");
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  return lines.map((line) => ({ type: "log", line }));
}
