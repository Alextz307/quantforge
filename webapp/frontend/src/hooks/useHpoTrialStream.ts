import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TrialRow } from "@/api/hpo";
import { hpoStreamUrl } from "@/api/hpo";
import { queryKeys } from "@/api/queryKeys";
import { useEventStream, type ConnectionState } from "@/hooks/useEventStream";

interface TrialFrame {
  type: "trial";
  trial: TrialRow;
}

export interface HpoTrialStreamSnapshot {
  trials: readonly TrialRow[];
  connection: ConnectionState;
}

/**
 * WebSocket subscription to /api/hpo/{wire_id}/stream.
 *
 * The backend replays all existing trials on connect (filtered by
 * ``afterTrial`` when set) and then live-tails ``trials.jsonl`` for any
 * new lines. Works uniformly across top-level webapp tune jobs, nested
 * HPO studies inside a running study job, and CLI-launched studies —
 * the source of truth is the on-disk file, not an in-process broker.
 *
 * Always-enabled by design: completed studies see no new frames, which
 * costs ~1s mtime polling on a static file. Earlier gating on
 * ``live_job_id`` hid streams for nested + CLI studies that have no
 * webapp job row but are still writing trials.
 */
export function useHpoTrialStream(wireId: string, afterTrial?: number): HpoTrialStreamSnapshot {
  const qc = useQueryClient();
  const [trials, setTrials] = useState<TrialRow[]>([]);

  const { connection } = useEventStream<TrialFrame>({
    url: hpoStreamUrl(wireId, afterTrial),
    parseFrame,
    onFrame: (frame) => {
      setTrials((prev) => {
        const next = mergeTrial(prev, frame.trial);
        qc.setQueryData<TrialRow[]>(queryKeys.hpoTrials(wireId), next);
        return next;
      });
    },
  });
  return { trials, connection };
}

/**
 * Append-fast-path merge: Optuna trials arrive in monotone ``number`` order
 * during a normal live run, so the common case is a pure push (O(1)). The
 * dup-scan and sort branches handle replay / retry frames that arrive out
 * of order.
 */
function mergeTrial(prev: TrialRow[], next: TrialRow): TrialRow[] {
  const last = prev[prev.length - 1];
  if (last === undefined || next.number > last.number) return [...prev, next];
  if (prev.some((t) => t.number === next.number)) return prev;
  return [...prev, next].sort((a, b) => a.number - b.number);
}

function parseFrame(raw: string): TrialFrame | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;
  if (obj.type !== "trial") return null;
  const trial = obj.trial;
  if (!trial || typeof trial !== "object") return null;
  // Backend serialises through pydantic; runtime guard checks the few fields
  // we depend on for ordering / display, then trusts the schema for the rest.
  const t = trial as Record<string, unknown>;
  if (typeof t.number !== "number" || typeof t.state !== "string") return null;
  return { type: "trial", trial: trial as TrialRow };
}
