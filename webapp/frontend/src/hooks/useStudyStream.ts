import { useQueryClient } from "@tanstack/react-query";
import type { StudyDetail } from "@/api/studies";
import { studyStreamUrl } from "@/api/studies";
import { queryKeys } from "@/api/queryKeys";
import { useEventStream, type ConnectionState } from "@/hooks/useEventStream";

export interface StudyStreamSnapshot {
  connection: ConnectionState;
}

/**
 * WebSocket subscription to /api/studies/{name}/stream.
 *
 * Each frame is a full ``StudyDetail`` snapshot the backend re-emits on
 * every ``study_state.json`` mtime bump. We mirror it into the static
 * ``useStudy`` cache so any sibling reader picks up the new state on
 * its next render (TanStack Query structural equality dedupes no-op
 * snapshots so the dependent components don't churn).
 */
export function useStudyStream(name: string, enabled: boolean = true): StudyStreamSnapshot {
  const qc = useQueryClient();
  const { connection } = useEventStream<StudyDetail>({
    url: studyStreamUrl(name),
    parseFrame,
    enabled,
    onFrame: (detail) => {
      qc.setQueryData<StudyDetail>(queryKeys.study(name), detail);
    },
  });
  return { connection };
}

function parseFrame(raw: string): StudyDetail | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;
  // Runtime guard: trust the Pydantic schema for the shape, sanity-check
  // the load-bearing fields the UI keys off so a stray non-StudyDetail
  // frame doesn't poison the cache.
  if (typeof obj.name !== "string" || typeof obj.total_legs !== "number") return null;
  if (!Array.isArray(obj.legs)) return null;
  return parsed as StudyDetail;
}
