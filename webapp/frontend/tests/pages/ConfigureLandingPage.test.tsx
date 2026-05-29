import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ROUTES } from "@/lib/routes";
import { ConfigureLandingPage } from "@/pages/ConfigureLandingPage";
import { renderWithProviders } from "../util/render";

describe("ConfigureLandingPage", () => {
  it("offers a deployment entry that opens the deploy picker", () => {
    renderWithProviders(<ConfigureLandingPage />);
    const link = screen.getByRole("link", { name: /New deployment/i });
    expect(link).toHaveAttribute("href", `${ROUTES.deployments}?new=1`);
  });
});
