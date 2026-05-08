import { useEffect, useRef } from "react";
import { cn } from "@/lib/cn";
import { ConnectionIndicator } from "@/components/ConnectionIndicator";
import type { ConnectionState } from "@/hooks/useEventStream";

interface LogPaneProps {
  lines: readonly string[];
  connection?: ConnectionState | undefined;
  emptyMessage?: string | undefined;
  className?: string | undefined;
}

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
      {connection !== undefined && <ConnectionIndicator state={connection} />}
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
