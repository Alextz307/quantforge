import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { StudyDetail } from "@/api/studies";
import { queryKeys } from "@/api/queryKeys";
import { useStudyStream } from "@/hooks/useStudyStream";
import { installMockWebSocket, MockWebSocket } from "../util/mockWebSocket";

const STUDY_NAME = "live_study";
const INITIAL_TOTAL = 2;
const INITIAL_COMPLETED = 0;
const NEXT_COMPLETED = 1;

installMockWebSocket();

afterEach(() => {
  vi.restoreAllMocks();
});

function makeDetail(completedLegs: number): StudyDetail {
  return {
    name: STUDY_NAME,
    spec_name: "tiny",
    spec_hash: "deadbeef",
    started_at: "2026-05-01T00:00:00Z",
    total_legs: INITIAL_TOTAL,
    completed_legs: completedLegs,
    completion_pct: (completedLegs / INITIAL_TOTAL) * 100,
    cross_strategy_compares_done: [],
    has_consolidated_report: false,
    legs: [],
  };
}

function buildWrapper(qc: QueryClient): (props: { children: ReactNode }) => ReactNode {
  return function Wrapper({ children }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

describe("useStudyStream", () => {
  it("does not connect when enabled=false", () => {
    const qc = new QueryClient();
    renderHook(() => useStudyStream(STUDY_NAME, false), { wrapper: buildWrapper(qc) });
    expect(MockWebSocket.instances.length).toBe(0);
  });

  it("mirrors arriving snapshots into the useStudy cache", async () => {
    const qc = new QueryClient();
    renderHook(() => useStudyStream(STUDY_NAME), { wrapper: buildWrapper(qc) });
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");

    act(() => {
      ws.triggerOpen();
      ws.triggerMessage(makeDetail(INITIAL_COMPLETED));
      ws.triggerMessage(makeDetail(NEXT_COMPLETED));
    });

    const cached = qc.getQueryData<StudyDetail>(queryKeys.study(STUDY_NAME));
    expect(cached?.completed_legs).toBe(NEXT_COMPLETED);
  });

  it("rejects malformed frames", async () => {
    const qc = new QueryClient();
    renderHook(() => useStudyStream(STUDY_NAME), { wrapper: buildWrapper(qc) });
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");

    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "log", line: "not a study detail" });
      ws.triggerMessage({ partial: "no name field" });
    });

    expect(qc.getQueryData<StudyDetail>(queryKeys.study(STUDY_NAME))).toBeUndefined();
  });
});
