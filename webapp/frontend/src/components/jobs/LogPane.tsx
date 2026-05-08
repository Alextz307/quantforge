import { useEffect, useRef } from "react";
import { cn } from "@/lib/cn";
import type { ConnectionState } from "@/hooks/useEventStream";

interface LogPaneProps {
  lines: readonly string[];
  connection?: ConnectionState | undefined;
  emptyMessage?: string | undefined;
  className?: string | undefined;
}

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  open: "Streaming",
  closed: "Disconnected",
  error: "Connection lost",
};

export function LogPane({
  lines,
  connection,
  emptyMessage = "No log output yet.",
  className,
}: LogPaneProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const previousLength = useRef(0);

  useEffect(() => {
    // Auto-scroll only when new lines arrive — preserves the user's
    // scroll position if they've manually scrolled up to read history.
    if (lines.length > previousLength.current) {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    }
    previousLength.current = lines.length;
  }, [lines]);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {connection !== undefined && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            data-testid="log-connection"
            data-state={connection}
            className={cn(
              "inline-block h-2 w-2 rounded-full",
              connection === "open" && "bg-emerald-500",
              connection === "connecting" && "bg-amber-500 animate-pulse",
              connection === "closed" && "bg-slate-400",
              connection === "error" && "bg-rose-500",
            )}
          />
          {CONNECTION_LABEL[connection]}
        </div>
      )}
      <div
        ref={scrollRef}
        data-testid="log-pane"
        className="h-72 overflow-auto rounded border bg-slate-950 p-3 font-mono text-xs text-slate-100"
      >
        {lines.length === 0 ? (
          <p className="text-slate-400">{emptyMessage}</p>
        ) : (
          lines.map((line, idx) => (
            <div key={idx} className="whitespace-pre-wrap break-words">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
