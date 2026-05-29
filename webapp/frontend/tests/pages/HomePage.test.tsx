import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HomePage } from "@/pages/HomePage";
import { ROUTES } from "@/lib/routes";
import { renderWithProviders } from "../util/render";

describe("HomePage", () => {
  it("renders the brand heading + tagline", () => {
    renderWithProviders(<HomePage />);
    expect(screen.getByRole("heading", { name: /QuantForge/i })).toBeInTheDocument();
    expect(screen.getByText(/anti-leakage/i)).toBeInTheDocument();
  });

  it("links each card to its section", () => {
    renderWithProviders(<HomePage />);
    const hrefs = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    const expected: readonly string[] = [
      ROUTES.configure,
      ROUTES.jobs,
      ROUTES.runs,
      ROUTES.studies,
      ROUTES.hpo,
      ROUTES.comparisons,
      ROUTES.holdout,
      ROUTES.deployments,
    ];
    for (const href of expected) {
      expect(hrefs).toContain(href);
    }
  });
});
