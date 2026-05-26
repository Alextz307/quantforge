import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { StudySpecFormatHelp } from "@/components/StudySpecFormatHelp";
import { API_PATHS, toMswPath } from "@/api/paths";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

function mockSchema() {
  server.use(
    http.get(toMswPath(API_PATHS.studySpecSchema), () =>
      HttpResponse.json({
        properties: {
          name: { type: "string", description: "Slug for artifacts." },
          legs: { type: "array", description: "Per-strategy legs." },
        },
        required: ["name", "legs"],
        $defs: {
          StudyLeg: {
            properties: {
              strategy: { type: "string", description: "Registered strategy name." },
              universes: { type: "array", description: "Slugs of universes." },
            },
            required: ["strategy", "universes"],
          },
        },
      }),
    ),
  );
}

function mockUniverses() {
  server.use(
    http.get(toMswPath(API_PATHS.configs), ({ params }) =>
      params.kind === "universe"
        ? HttpResponse.json([{ name: "spy_daily_5y" }, { name: "qqq_daily_5y" }])
        : HttpResponse.json([]),
    ),
  );
}

describe("StudySpecFormatHelp", () => {
  it("renders schema-derived rows with descriptions", async () => {
    mockSchema();
    mockUniverses();
    renderWithProviders(<StudySpecFormatHelp onInsertLeg={() => undefined} />);

    // The schema-derived path appears in both the table and the description
    // list — getAllByText asserts that both rendered.
    expect((await screen.findAllByText("legs[*].strategy")).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/Slug for artifacts/i)).toBeInTheDocument();
    expect(screen.getByText(/Registered strategy name/i)).toBeInTheDocument();
  });

  it("Insert leg button calls onInsertLeg with the leg snippet", async () => {
    mockSchema();
    mockUniverses();
    const handle = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<StudySpecFormatHelp onInsertLeg={handle} />);

    await user.click(screen.getByRole("button", { name: /Insert leg template/i }));
    expect(handle).toHaveBeenCalled();
    const snippet = handle.mock.calls[0]?.[0] as string | undefined;
    expect(snippet).toBeDefined();
    expect(snippet).toContain("strategy:");
    expect(snippet).toContain("universes:");
  });

  it("Browse universes toggles a list of universe slugs", async () => {
    mockSchema();
    mockUniverses();
    const user = userEvent.setup();
    renderWithProviders(<StudySpecFormatHelp onInsertLeg={() => undefined} />);

    await user.click(screen.getByRole("button", { name: /Browse universes/i }));
    expect(await screen.findByText("spy_daily_5y")).toBeInTheDocument();
    expect(screen.getByText("qqq_daily_5y")).toBeInTheDocument();
  });
});
