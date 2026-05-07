import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { useJobStream } from "@/hooks/useJobStream";
import { queryKeys } from "@/api/queryKeys";
import type { JobRow } from "@/api/jobs";

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  url: string;
  readyState = 0;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent<string>) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  triggerOpen() {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  triggerMessage(payload: object) {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(payload) }));
  }

  triggerClose() {
    this.readyState = 3;
    this.onclose?.(new CloseEvent("close"));
  }

  close() {
    this.triggerClose();
  }
}

const realWebSocket = globalThis.WebSocket;

beforeAll(() => {
  MockWebSocket.instances = [];
  // The hook only uses WebSocket as a constructor at runtime; vitest's
  // jsdom environment doesn't ship one. Cast through unknown to satisfy
  // the WebSocket constructor type without dragging in a full polyfill.
  Object.defineProperty(globalThis, "WebSocket", {
    value: MockWebSocket,
    writable: true,
    configurable: true,
  });
});

afterAll(() => {
  Object.defineProperty(globalThis, "WebSocket", {
    value: realWebSocket,
    writable: true,
    configurable: true,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

function mockLogResponse(body: string): void {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(body, { status: 200, headers: { "Content-Type": "text/plain" } }),
  );
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  client.setQueryData<JobRow>(queryKeys.job("job-x"), {
    id: "job-x",
    user_id: 1,
    kind: "run",
    status: "running",
    started_at: "2026-05-07T10:00:00Z",
    finished_at: null,
    exit_code: null,
    experiment_id: null,
    log_path: "/tmp/job-x.log",
    pid: 1,
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useJobStream", () => {
  it("appends log frames to the logs array and reports the connection state", async () => {
    MockWebSocket.instances = [];
    const { result } = renderHook(() => useJobStream("job-x", "running"), { wrapper });

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");

    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "log", line: "starting" });
      ws.triggerMessage({ type: "log", line: "training fold 0" });
    });

    expect(result.current.logs).toEqual(["starting", "training fold 0"]);
    expect(result.current.connection).toBe("open");
  });

  it("backfills logs from the persisted file when the job is already terminal", async () => {
    MockWebSocket.instances = [];
    mockLogResponse("first line\nsecond line\n");
    const { result } = renderHook(() => useJobStream("job-x", "completed"), { wrapper });
    expect(result.current.connection).toBe("closed");
    expect(MockWebSocket.instances.length).toBe(0);
    await waitFor(() => {
      expect(result.current.logs).toEqual(["first line", "second line"]);
    });
  });

  it("closes the socket when a terminal status frame arrives", async () => {
    MockWebSocket.instances = [];
    renderHook(() => useJobStream("job-x", "running"), { wrapper });
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "status", status: "completed", experiment_id: "exp_x" });
    });
    expect(ws.readyState).toBe(3);
  });
});
