import { cn } from "@/lib/cn";
import type { ConnectionState } from "@/hooks/useEventStream";

const STATE_META: Record<ConnectionState, { label: string; dot: string }> = {
  connecting: { label: "Connecting...", dot: "bg-amber-500 animate-pulse" },
  open: { label: "Streaming", dot: "bg-emerald-500" },
  closed: { label: "Disconnected", dot: "bg-slate-400" },
  error: { label: "Connection lost", dot: "bg-rose-500" },
};

interface ConnectionIndicatorProps {
  state: ConnectionState;
  className?: string | undefined;
}

export function ConnectionIndicator({ state, className }: ConnectionIndicatorProps) {
  const meta = STATE_META[state];
  return (
    <div className={cn("flex items-center gap-2 text-xs text-muted-foreground", className)}>
      <span
        data-testid="connection-indicator"
        data-state={state}
        className={cn("inline-block h-2 w-2 rounded-full", meta.dot)}
      />
      {meta.label}
    </div>
  );
}
