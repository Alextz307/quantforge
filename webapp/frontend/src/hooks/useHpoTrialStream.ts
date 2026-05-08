import { useMemo } from "react";
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
 * WebSocket subscription to /api/hpo/{name}/stream.
 *
 * The backend replays all existing trials on connect (filtered by
 * ``afterTrial`` when set) and then forwards live broker frames. We
 * append-merge into the static ``useHpoTrials`` cache so a brief unmount
 * (e.g. tab switch) doesn't drop already-arrived rows; the in-memory
 * ``trials`` array is the authoritative ordered view for the page.
 *
 * ``isLive`` defers to the caller to know when the page is in
 * live-monitor mode (study.live_job_id != null) — keeps this hook a
 * pure WS consumer with no own data fetching.
 */
export function useHpoTrialStream(
  name: string,
  isLive: boolean,
  afterTrial?: number,
): HpoTrialStreamSnapshot {
  const qc = useQueryClient();
  const { frames, connection } = useEventStream<TrialFrame>({
    url: hpoStreamUrl(name, afterTrial),
    parseFrame,
    enabled: isLive,
    onFrame: (frame) => {
      qc.setQueryData<TrialRow[]>(queryKeys.hpoTrials(name), (prev) => {
        if (!prev) return [frame.trial];
        if (prev.some((t) => t.number === frame.trial.number)) return prev;
        return [...prev, frame.trial].sort((a, b) => a.number - b.number);
      });
    },
  });
  const trials = useMemo(() => frames.map((f) => f.trial), [frames]);
  return { trials, connection };
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
