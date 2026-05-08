import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJobStream } from "@/hooks/useJobStream";
import { queryKeys } from "@/api/queryKeys";
import type { JobRow } from "@/api/jobs";
import { installMockWebSocket, MockWebSocket } from "../util/mockWebSocket";

installMockWebSocket();

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
    mockLogResponse("first line\nsecond line\n");
    const { result } = renderHook(() => useJobStream("job-x", "completed"), { wrapper });
    expect(result.current.connection).toBe("closed");
    expect(MockWebSocket.instances.length).toBe(0);
    await waitFor(() => {
      expect(result.current.logs).toEqual(["first line", "second line"]);
    });
  });

  it("closes the socket when a terminal status frame arrives", async () => {
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
