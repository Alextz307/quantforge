import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useEventStream } from "@/hooks/useEventStream";
import { installMockWebSocket, MockWebSocket } from "../util/mockWebSocket";

interface PingFrame {
  type: "ping";
  seq: number;
}

const TEST_URL = "ws://test.local/stream";
const BACKFILL_URL = "/api/test/backfill";

installMockWebSocket();

afterEach(() => {
  vi.restoreAllMocks();
});

function parsePing(raw: string): PingFrame | null {
  try {
    const obj = JSON.parse(raw) as Record<string, unknown>;
    if (obj.type !== "ping" || typeof obj.seq !== "number") return null;
    return { type: "ping", seq: obj.seq };
  } catch {
    return null;
  }
}

describe("useEventStream", () => {
  it("invokes onFrame for each parsed frame in order and reports the open connection", async () => {
    const onFrame = vi.fn();
    const { result } = renderHook(() =>
      useEventStream<PingFrame>({ url: TEST_URL, parseFrame: parsePing, onFrame }),
    );
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");

    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "ping", seq: 1 });
      ws.triggerMessage({ type: "ping", seq: 2 });
    });

    expect(onFrame).toHaveBeenCalledTimes(2);
    expect(onFrame).toHaveBeenNthCalledWith(1, { type: "ping", seq: 1 });
    expect(onFrame).toHaveBeenNthCalledWith(2, { type: "ping", seq: 2 });
    expect(result.current.connection).toBe("open");
  });

  it("calls onFrame for duplicate frames (consumer-side dedup is not the hook's concern)", async () => {
    const onFrame = vi.fn();
    renderHook(() => useEventStream<PingFrame>({ url: TEST_URL, parseFrame: parsePing, onFrame }));
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "ping", seq: 1 });
      ws.triggerMessage({ type: "ping", seq: 1 });
    });
    expect(onFrame).toHaveBeenCalledTimes(2);
  });

  it("closes the socket when shouldClose returns true and freezes the connection state", async () => {
    const { result } = renderHook(() =>
      useEventStream<PingFrame>({
        url: TEST_URL,
        parseFrame: parsePing,
        shouldClose: (f) => f.seq >= 2,
      }),
    );
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "ping", seq: 1 });
      ws.triggerMessage({ type: "ping", seq: 2 });
    });
    expect(ws.readyState).toBe(3);
    await waitFor(() => {
      expect(result.current.connection).toBe("closed");
    });
  });

  it("runs backfill once when enabled is false and never opens a socket", async () => {
    const onFrame = vi.fn();
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ type: "ping", seq: 7 }), { status: 200 }));
    const { result } = renderHook(() =>
      useEventStream<PingFrame>({
        url: TEST_URL,
        parseFrame: parsePing,
        enabled: false,
        onFrame,
        backfillUrl: BACKFILL_URL,
        backfillParse: (text) => {
          const f = parsePing(text);
          return f ? [f] : [];
        },
      }),
    );

    await waitFor(() => {
      expect(onFrame).toHaveBeenCalledWith({ type: "ping", seq: 7 });
    });
    expect(MockWebSocket.instances.length).toBe(0);
    expect(result.current.connection).toBe("closed");
    expect(fetchSpy).toHaveBeenCalledWith(
      BACKFILL_URL,
      expect.objectContaining({ credentials: "include" }),
    );
  });
});
