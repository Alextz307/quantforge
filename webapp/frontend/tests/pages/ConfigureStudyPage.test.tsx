import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Route, Routes } from "react-router-dom";
import { ConfigureStudyPage } from "@/pages/ConfigureStudyPage";
import { API_PATHS, toMswPath } from "@/api/paths";
import { ROUTES } from "@/lib/routes";
import { server } from "../msw/server";
import { renderWithProviders } from "../util/render";

const SPEC_NAME = "main_study";

function mockStudySpecsList() {
  server.use(
    http.get(toMswPath(API_PATHS.configs), ({ params }) => {
      if (params.kind === "study") return HttpResponse.json([{ name: SPEC_NAME }]);
      if (params.kind === "universe") return HttpResponse.json([{ name: "spy_daily_5y" }]);
      return HttpResponse.json([]);
    }),
  );
}

function mockUploadsEmpty() {
  server.use(http.get(toMswPath(API_PATHS.studyUploads), () => HttpResponse.json([])));
}

function mockSchema() {
  server.use(
    http.get(toMswPath(API_PATHS.studySpecSchema), () =>
      HttpResponse.json({
        properties: {
          name: { type: "string", description: "Study name slug." },
          legs: { type: "array", description: "Per-strategy legs." },
        },
        required: ["name", "legs"],
        $defs: {
          StudyLeg: {
            properties: {
              strategy: { type: "string", description: "Registered strategy name." },
            },
            required: ["strategy"],
          },
        },
      }),
    ),
  );
}

describe("ConfigureStudyPage", () => {
  it("submits a library spec and navigates to the job", async () => {
    mockStudySpecsList();
    mockUploadsEmpty();
    mockSchema();
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configureStudy} element={<ConfigureStudyPage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configureStudy] },
    );

    const select = await screen.findByLabelText(/Library spec/i);
    await user.selectOptions(select, SPEC_NAME);
    await user.click(screen.getByRole("button", { name: /Launch study/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("blocks library-tab submit when no spec is selected", async () => {
    mockStudySpecsList();
    mockUploadsEmpty();
    mockSchema();
    const user = userEvent.setup();
    renderWithProviders(<ConfigureStudyPage />);

    await screen.findByLabelText(/Library spec/i);
    await user.click(screen.getByRole("button", { name: /Launch study/i }));

    expect(await screen.findByText(/Pick a study spec/i)).toBeInTheDocument();
  });

  it("surfaces an empty-state when no library specs are registered", async () => {
    server.use(
      http.get(toMswPath(API_PATHS.configs), () => HttpResponse.json([])),
      http.get(toMswPath(API_PATHS.studyUploads), () => HttpResponse.json([])),
    );
    mockSchema();
    renderWithProviders(<ConfigureStudyPage />);

    expect(await screen.findByText(/No study specs found/i)).toBeInTheDocument();
  });

  it("surfaces backend 422 errors inline", async () => {
    mockStudySpecsList();
    mockUploadsEmpty();
    mockSchema();
    server.use(
      http.post(API_PATHS.jobs, () =>
        HttpResponse.json(
          {
            detail: [
              {
                loc: ["study_payload", "spec_name"],
                msg: "study 'main' is already running under job 'abc'",
                type: "value_error",
              },
            ],
          },
          { status: 422 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigureStudyPage />);

    const select = await screen.findByLabelText(/Library spec/i);
    await user.selectOptions(select, SPEC_NAME);
    await user.click(screen.getByRole("button", { name: /Launch study/i }));

    expect(await screen.findByText(/already running/i)).toBeInTheDocument();
  });

  it("switches to the New tab and shows the annotated skeleton", async () => {
    mockStudySpecsList();
    mockUploadsEmpty();
    mockSchema();
    const user = userEvent.setup();
    renderWithProviders(<ConfigureStudyPage />);

    await user.click(screen.getByRole("tab", { name: /New spec/i }));

    const editor = await screen.findByTestId<HTMLTextAreaElement>("monaco-editor");
    expect(editor.value).toContain("name: my_first_study");
    expect(editor.value).toContain("legs:");
    expect(editor.value).toContain("AdaptiveBollinger");
  });

  it("Save & launch round trip creates an upload then submits the job", async () => {
    mockStudySpecsList();
    mockUploadsEmpty();
    mockSchema();
    server.use(
      // /api/configs/study_spec/validate — always returns valid for this test.
      http.post(toMswPath(API_PATHS.studySpecValidate), () =>
        HttpResponse.json({ valid: true, errors: [] }),
      ),
      // POST /api/configs/study/uploads — record the save.
      http.post(toMswPath(API_PATHS.studyUploads), async ({ request }) => {
        const body = (await request.json()) as { slug: string; yaml: string };
        return HttpResponse.json(
          {
            slug: body.slug,
            yaml: body.yaml,
            created_at: "2026-05-27T00:00:00Z",
            updated_at: "2026-05-27T00:00:00Z",
            owner_user_id: 1,
            owner_username: "alex",
          },
          { status: 201 },
        );
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configureStudy} element={<ConfigureStudyPage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configureStudy] },
    );

    await user.click(screen.getByRole("tab", { name: /New spec/i }));
    const slugInput = await screen.findByLabelText(/Slug/i);
    await user.type(slugInput, "my_new_study");

    await user.click(screen.getByRole("button", { name: /Save & launch/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("Save (without launch) flips the mode to the uploads tab", async () => {
    mockStudySpecsList();
    server.use(
      // Start with empty uploads, then return the new one after save.
      http.get(toMswPath(API_PATHS.studyUploads), () => HttpResponse.json([])),
      http.post(toMswPath(API_PATHS.studySpecValidate), () =>
        HttpResponse.json({ valid: true, errors: [] }),
      ),
      http.post(toMswPath(API_PATHS.studyUploads), async ({ request }) => {
        const body = (await request.json()) as { slug: string; yaml: string };
        return HttpResponse.json(
          {
            slug: body.slug,
            yaml: body.yaml,
            created_at: "2026-05-27T00:00:00Z",
            updated_at: "2026-05-27T00:00:00Z",
            owner_user_id: 1,
            owner_username: "alex",
          },
          { status: 201 },
        );
      }),
    );
    mockSchema();
    const user = userEvent.setup();
    renderWithProviders(<ConfigureStudyPage />);

    await user.click(screen.getByRole("tab", { name: /New spec/i }));
    const slugInput = await screen.findByLabelText(/Slug/i);
    await user.type(slugInput, "draft_study");
    await user.click(screen.getByRole("button", { name: /^Save$/i }));

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: /My uploads/i })).toHaveAttribute(
        "aria-selected",
        "true",
      );
    });
  });

  it("library-collision (409) surfaces a descriptive error", async () => {
    mockStudySpecsList();
    mockUploadsEmpty();
    mockSchema();
    server.use(
      http.post(toMswPath(API_PATHS.studySpecValidate), () =>
        HttpResponse.json({ valid: true, errors: [] }),
      ),
      http.post(toMswPath(API_PATHS.studyUploads), () =>
        HttpResponse.json({ detail: "slug 'main_study' shadows a library spec" }, { status: 409 }),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<ConfigureStudyPage />);

    await user.click(screen.getByRole("tab", { name: /New spec/i }));
    await user.type(await screen.findByLabelText(/Slug/i), "main_study");
    await user.click(screen.getByRole("button", { name: /Save & launch/i }));

    expect(await screen.findByText(/shadows a library spec/i)).toBeInTheDocument();
    expect(screen.queryByText("job detail page")).not.toBeInTheDocument();
  });
});
