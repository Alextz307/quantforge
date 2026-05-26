import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { createElement } from "react";
import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { server } from "./msw/server";

// Plotly cannot mount in jsdom (no canvas/WebGL). Replace `react-plotly.js/factory`
// with a stub that surfaces trace count via a data attribute so tests can assert
// the chart received the expected number of series without rendering SVG.
vi.mock("react-plotly.js/factory", () => ({
  default: () =>
    function MockPlot(props: { data?: ReadonlyArray<{ name?: string }> }) {
      return createElement("div", {
        "data-testid": "plotly-plot",
        "data-trace-count": props.data?.length ?? 0,
        "data-trace-names": (props.data ?? []).map((d) => d.name ?? "").join(","),
      });
    },
}));

// Monaco's WebGL + worker pipeline can't mount in jsdom. Stand in with a plain
// <textarea> so YAML editor tests can still exercise text changes via fireEvent.
vi.mock("@monaco-editor/react", () => ({
  default: function MockEditor(props: {
    value?: string;
    onChange?: (next: string | undefined) => void;
    options?: { readOnly?: boolean };
  }) {
    return createElement("textarea", {
      "data-testid": "monaco-editor",
      value: props.value ?? "",
      readOnly: props.options?.readOnly ?? false,
      onChange: (e: { target: { value: string } }) => props.onChange?.(e.target.value),
    });
  },
}));

beforeAll(() => {
  server.listen({ onUnhandledRequest: "error" });
});

afterEach(() => {
  cleanup();
  server.resetHandlers();
});

afterAll(() => {
  server.close();
});
