import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { JobRow, JobStatus } from "@/api/jobs";
import { isTerminalStatus, jobLogDownloadUrl, jobStreamUrl } from "@/api/jobs";
import { queryKeys } from "@/api/queryKeys";

export type ConnectionState = "connecting" | "open" | "closed" | "error";

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

const RECONNECT_DELAYS_MS = [250, 500, 1000] as const;

/**
 * Native WebSocket subscription to /api/jobs/{id}/stream with bounded
 * exponential-backoff reconnect.
 *
 * Side-effects on status frames:
 *   - Patches the per-job cache via setQueryData (cheaper than refetching
 *     the row we already received from the broker). The updater short-
 *     circuits on identical status to avoid waking observers on no-op
 *     heartbeats.
 *   - Invalidates the jobs list only on terminal status — the running
 *     ``useJobs`` poll already keeps the list fresh while jobs are open.
 *   - On a terminal-completed status with an experiment_id, invalidates
 *     the runs list so the freshly-finished run appears in /runs.
 */
export function useJobStream(jobId: string, initialStatus: JobStatus): JobStreamSnapshot {
  const qc = useQueryClient();
  const [logs, setLogs] = useState<string[]>([]);
  const [connection, setConnection] = useState<ConnectionState>(
    isTerminalStatus(initialStatus) ? "closed" : "connecting",
  );
  // Latest known status survives reconnects so a re-open after terminal stays closed.
  const latestStatus = useRef<JobStatus>(initialStatus);
  const retryAttempt = useRef<number>(0);
  const reconnectTimer = useRef<number | null>(null);
  const ws = useRef<WebSocket | null>(null);

  const handleStatus = useCallback(
    (frame: StatusFrame) => {
      latestStatus.current = frame.status;
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
          void qc.invalidateQueries({ queryKey: queryKeys.runs });
        }
      }
    },
    [jobId, qc],
  );

  useEffect(() => {
    if (isTerminalStatus(initialStatus)) {
      // Terminal at mount: backfill from the persisted log file.
      setConnection("closed");
      const controller = new AbortController();
      void (async () => {
        try {
          const resp = await fetch(jobLogDownloadUrl(jobId), {
            credentials: "include",
            signal: controller.signal,
          });
          if (!resp.ok) return;
          const text = await resp.text();
          const lines = text.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          setLogs(lines);
        } catch {
          // abort or network failure
        }
      })();
      return () => {
        controller.abort();
      };
    }

    let disposed = false;

    const connect = () => {
      if (disposed) return;
      const socket = new WebSocket(jobStreamUrl(jobId));
      ws.current = socket;
      setConnection("connecting");

      socket.onopen = () => {
        retryAttempt.current = 0;
        setConnection("open");
      };

      socket.onmessage = (event: MessageEvent<string>) => {
        const frame = parseFrame(event.data);
        if (!frame) return;
        if (frame.type === "log") {
          setLogs((prev) => [...prev, frame.line]);
        } else {
          handleStatus(frame);
          if (isTerminalStatus(frame.status)) {
            socket.close();
          }
        }
      };

      socket.onerror = () => {
        setConnection("error");
      };

      socket.onclose = () => {
        ws.current = null;
        if (disposed) return;
        if (isTerminalStatus(latestStatus.current)) {
          setConnection("closed");
          return;
        }
        const delay = RECONNECT_DELAYS_MS[retryAttempt.current];
        if (delay === undefined) {
          setConnection("error");
          return;
        }
        retryAttempt.current += 1;
        reconnectTimer.current = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.onmessage = null;
        ws.current.onerror = null;
        ws.current.onopen = null;
        ws.current.close();
        ws.current = null;
      }
    };
  }, [jobId, initialStatus, handleStatus]);

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
