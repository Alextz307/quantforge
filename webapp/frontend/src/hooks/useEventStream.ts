import { useEffect, useRef, useState } from "react";

export type ConnectionState = "connecting" | "open" | "closed" | "error";

export interface UseEventStreamOptions<TFrame> {
  url: string;
  parseFrame: (raw: string) => TFrame | null;
  /** When false, skips opening the socket and (if a backfill URL is set) runs the backfill once. */
  enabled?: boolean;
  /** Called for every parsed frame, including those replayed via backfill. */
  onFrame?: (frame: TFrame) => void;
  /** Returning true closes the socket after the frame; surface terminal-status frames here. */
  shouldClose?: (frame: TFrame) => boolean;
  /** GET endpoint queried once when ``enabled === false`` to seed historical state. */
  backfillUrl?: string;
  /** Parses the backfill response body into frames; defaults to no-op (empty array). */
  backfillParse?: (text: string) => readonly TFrame[];
}

export interface EventStreamSnapshot {
  connection: ConnectionState;
}

const RECONNECT_DELAYS_MS = [250, 500, 1000] as const;

/**
 * Native WebSocket subscription with bounded exponential-backoff reconnect,
 * terminal-state freeze, and an optional one-shot HTTP backfill for sockets
 * that were already closed at mount time.
 *
 * Frames are NOT retained inside the hook - consumers are responsible for
 * accumulating any derived state in their own ``onFrame`` handler. This
 * keeps storage append-only at the consumer site (no per-frame filter+map
 * over a shadow array) and avoids holding two copies of the data.
 */
export function useEventStream<TFrame>(opts: UseEventStreamOptions<TFrame>): EventStreamSnapshot {
  const {
    url,
    parseFrame,
    enabled = true,
    onFrame,
    shouldClose,
    backfillUrl,
    backfillParse,
  } = opts;

  const [connection, setConnection] = useState<ConnectionState>(enabled ? "connecting" : "closed");
  // Once shouldClose() fires (or the hook mounted disabled), the reconnect
  // ladder must NOT bring the socket back up.
  const frozen = useRef<boolean>(!enabled);
  const retryAttempt = useRef<number>(0);
  const reconnectTimer = useRef<number | null>(null);
  const ws = useRef<WebSocket | null>(null);

  // Fresh callbacks every render are normal - pin them in refs so the effect
  // body doesn't tear down the socket whenever a parent re-renders.
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;
  const shouldCloseRef = useRef(shouldClose);
  shouldCloseRef.current = shouldClose;
  const parseFrameRef = useRef(parseFrame);
  parseFrameRef.current = parseFrame;
  const backfillParseRef = useRef(backfillParse);
  backfillParseRef.current = backfillParse;

  useEffect(() => {
    if (!enabled) {
      frozen.current = true;
      setConnection("closed");
      if (!backfillUrl) return;
      const controller = new AbortController();
      void (async () => {
        try {
          const resp = await fetch(backfillUrl, {
            credentials: "include",
            signal: controller.signal,
          });
          if (!resp.ok) return;
          const text = await resp.text();
          const parsed = backfillParseRef.current?.(text) ?? [];
          for (const frame of parsed) onFrameRef.current?.(frame);
        } catch {
          // abort or network failure - no recovery; backfill is best-effort.
        }
      })();
      return () => {
        controller.abort();
      };
    }

    frozen.current = false;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      const socket = new WebSocket(url);
      ws.current = socket;
      setConnection("connecting");

      socket.onopen = () => {
        retryAttempt.current = 0;
        setConnection("open");
      };

      socket.onmessage = (event: MessageEvent<string>) => {
        const frame = parseFrameRef.current(event.data);
        if (!frame) return;
        onFrameRef.current?.(frame);
        if (shouldCloseRef.current?.(frame)) {
          frozen.current = true;
          socket.close();
        }
      };

      socket.onerror = () => {
        setConnection("error");
      };

      socket.onclose = () => {
        ws.current = null;
        if (disposed) return;
        if (frozen.current) {
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
  }, [url, enabled, backfillUrl]);

  return { connection };
}
