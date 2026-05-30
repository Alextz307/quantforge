import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SignalBadge } from "@/components/SignalBadge";

describe("SignalBadge", () => {
  it("renders a bare LONG pill for a directional positive signal", () => {
    render(<SignalBadge signal={1} />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveAttribute("data-signal-state", "long");
    expect(badge).toHaveTextContent("LONG");
    expect(badge).not.toHaveTextContent("x");
  });

  it("renders a bare SHORT pill for a directional negative signal", () => {
    render(<SignalBadge signal={-1} />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveAttribute("data-signal-state", "short");
    expect(badge).toHaveTextContent("SHORT");
  });

  it("shows the leverage multiplier for a leverage-kind long signal", () => {
    render(<SignalBadge signal={1.3853} kind="leverage" />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveAttribute("data-signal-state", "long");
    expect(badge).toHaveTextContent("LONG | 1.39x");
  });

  it("shows the leverage multiplier for a leverage-kind short signal", () => {
    render(<SignalBadge signal={-0.85} kind="leverage" />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveAttribute("data-signal-state", "short");
    expect(badge).toHaveTextContent("SHORT | 0.85x");
  });

  it("renders a FLAT pill for a zero signal", () => {
    render(<SignalBadge signal={0} />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveAttribute("data-signal-state", "flat");
    expect(badge).toHaveTextContent("FLAT");
  });

  it("renders the computing state while loading", () => {
    render(<SignalBadge signal={null} loading />);
    const badge = screen.getByTestId("signal-badge");
    expect(badge).toHaveAttribute("data-signal-state", "computing");
    expect(badge).toHaveTextContent("computing");
  });

  it("renders an unknown placeholder for a null signal", () => {
    render(<SignalBadge signal={null} />);
    expect(screen.getByTestId("signal-badge")).toHaveAttribute("data-signal-state", "unknown");
  });

  it("exposes the raw signed value in the title for finite signals", () => {
    render(<SignalBadge signal={1.2345} />);
    expect(screen.getByTestId("signal-badge")).toHaveAttribute("title", "1.2345");
  });
});
