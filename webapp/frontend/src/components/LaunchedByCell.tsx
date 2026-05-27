import type { ReactElement } from "react";

const SYSTEM_FALLBACK_LABEL = "system";

interface LaunchedByCellProps {
  username: string | null | undefined;
}

/**
 * Renders the "Launched by" column for artifact list pages.
 *
 * - When the backend returns a username (the artifact was launched through
 *   the webapp or backfilled), display it verbatim.
 * - When the backend returns `null` (CLI-launched / ownerless artifacts that
 *   weren't backfilled), fall back to a muted `"system"` label. The artifact
 *   is shared with every logged-in user; the label keeps the column honest
 *   about provenance without claiming a specific owner.
 */
export function LaunchedByCell({ username }: LaunchedByCellProps): ReactElement {
  if (username === null || username === undefined) {
    return <span className="text-muted-foreground italic">{SYSTEM_FALLBACK_LABEL}</span>;
  }
  return <span>{username}</span>;
}
