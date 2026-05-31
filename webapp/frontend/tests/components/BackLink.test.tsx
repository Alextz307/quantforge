import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { BackLink } from "@/components/BackLink";
import { ROUTER_FUTURE_FLAGS } from "../util/router";

type Entry = string | { pathname: string; state?: { from: string } };

function renderBackLink(entry: Entry) {
  return render(
    <MemoryRouter initialEntries={[entry]} future={ROUTER_FUTURE_FLAGS}>
      <BackLink to="/runs">All runs</BackLink>
    </MemoryRouter>,
  );
}

describe("BackLink", () => {
  it("links to the list with its label by default", () => {
    renderBackLink("/runs/abc");
    const link = screen.getByRole("link");

    expect(link).toHaveTextContent("All runs");
    expect(link).toHaveAttribute("href", "/runs");
  });

  it("returns to the stashed list url and keeps the label", () => {
    renderBackLink({ pathname: "/runs/abc", state: { from: "/runs?sort_by=sharpe_mean" } });
    const link = screen.getByRole("link");

    expect(link).toHaveTextContent("All runs");
    expect(link).toHaveAttribute("href", "/runs?sort_by=sharpe_mean");
  });

  it("shows a generic 'Back' when the stash points somewhere other than the list", () => {
    renderBackLink({ pathname: "/runs/abc", state: { from: "/runs/source-run" } });
    const link = screen.getByRole("link");

    expect(link).toHaveTextContent("Back");
    expect(link).not.toHaveTextContent("All runs");
    expect(link).toHaveAttribute("href", "/runs/source-run");
  });
});
