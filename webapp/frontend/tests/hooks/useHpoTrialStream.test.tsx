import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TrialRow } from "@/api/hpo";
import { queryKeys } from "@/api/queryKeys";
import { useHpoTrialStream } from "@/hooks/useHpoTrialStream";
import { installMockWebSocket, MockWebSocket } from "../util/mockWebSocket";

const STUDY_NAME = "live_study";
const TRIAL_NUMBER_FIRST = 0;
const TRIAL_NUMBER_SECOND = 1;

installMockWebSocket();

afterEach(() => {
  vi.restoreAllMocks();
});

function makeTrial(number: number): TrialRow {
  return {
    number,
    state: "COMPLETE",
    value: 0.5 + number * 0.1,
    params: { window: 30 + number },
    datetime_start: null,
    datetime_complete: null,
    experiment_id: null,
  };
}

function buildWrapper(qc: QueryClient): (props: { children: ReactNode }) => ReactNode {
  return function Wrapper({ children }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

describe("useHpoTrialStream", () => {
  it("appends arriving trials and seeds the trials cache append-merge", async () => {
    const qc = new QueryClient();
    const { result } = renderHook(() => useHpoTrialStream(STUDY_NAME), {
      wrapper: buildWrapper(qc),
    });
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");

    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "trial", trial: makeTrial(TRIAL_NUMBER_FIRST) });
      ws.triggerMessage({ type: "trial", trial: makeTrial(TRIAL_NUMBER_SECOND) });
    });

    expect(result.current.trials.map((t) => t.number)).toEqual([
      TRIAL_NUMBER_FIRST,
      TRIAL_NUMBER_SECOND,
    ]);
    const cached = qc.getQueryData<TrialRow[]>(queryKeys.hpoTrials(STUDY_NAME));
    expect(cached?.map((t) => t.number)).toEqual([TRIAL_NUMBER_FIRST, TRIAL_NUMBER_SECOND]);
  });

  it("ignores duplicate trial numbers in the cache merge", async () => {
    const qc = new QueryClient();
    qc.setQueryData<TrialRow[]>(queryKeys.hpoTrials(STUDY_NAME), [makeTrial(TRIAL_NUMBER_FIRST)]);
    renderHook(() => useHpoTrialStream(STUDY_NAME), { wrapper: buildWrapper(qc) });
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "trial", trial: makeTrial(TRIAL_NUMBER_FIRST) });
    });
    const cached = qc.getQueryData<TrialRow[]>(queryKeys.hpoTrials(STUDY_NAME));
    expect(cached?.map((t) => t.number)).toEqual([TRIAL_NUMBER_FIRST]);
  });

  it("rejects malformed frames", async () => {
    const qc = new QueryClient();
    const { result } = renderHook(() => useHpoTrialStream(STUDY_NAME), {
      wrapper: buildWrapper(qc),
    });
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });
    const ws = MockWebSocket.instances[0];
    if (!ws) throw new Error("WebSocket was never opened");
    act(() => {
      ws.triggerOpen();
      ws.triggerMessage({ type: "log", line: "not a trial" });
      ws.triggerMessage({ type: "trial", trial: { partial: "no number" } });
    });
    expect(result.current.trials).toEqual([]);
  });
});
