import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LogPane } from "@/components/jobs/LogPane";

describe("LogPane", () => {
  it("renders the empty-state message when no lines", () => {
    render(<LogPane lines={[]} />);
    expect(screen.getByText(/No log output yet\./i)).toBeInTheDocument();
  });

  it("renders the connection indicator with the matching data-state", () => {
    render(<LogPane lines={[]} connection="open" />);
    const indicator = screen.getByTestId("log-connection");
    expect(indicator.dataset.state).toBe("open");
    expect(screen.getByText(/Streaming/i)).toBeInTheDocument();
  });

  it("renders each log line as its own row", () => {
    render(<LogPane lines={["first", "second", "third"]} />);
    const pane = screen.getByTestId("log-pane");
    expect(pane.textContent).toContain("first");
    expect(pane.textContent).toContain("second");
    expect(pane.textContent).toContain("third");
    expect(pane.querySelectorAll("div").length).toBe(3);
  });
});
