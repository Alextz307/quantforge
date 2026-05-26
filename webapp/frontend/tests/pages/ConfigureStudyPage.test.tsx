import { screen } from "@testing-library/react";
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
      return HttpResponse.json([]);
    }),
  );
}

describe("ConfigureStudyPage", () => {
  it("submits a study payload and navigates to the job", async () => {
    mockStudySpecsList();
    const user = userEvent.setup();
    renderWithProviders(
      <Routes>
        <Route path={ROUTES.configureStudy} element={<ConfigureStudyPage />} />
        <Route path={ROUTES.jobDetail} element={<div>job detail page</div>} />
      </Routes>,
      { initialEntries: [ROUTES.configureStudy] },
    );

    const select = await screen.findByLabelText(/Spec/i);
    await user.selectOptions(select, SPEC_NAME);
    await user.click(screen.getByRole("button", { name: /Launch study/i }));

    expect(await screen.findByText("job detail page")).toBeInTheDocument();
  });

  it("blocks submit when no spec is selected", async () => {
    mockStudySpecsList();
    const user = userEvent.setup();
    renderWithProviders(<ConfigureStudyPage />);

    await screen.findByLabelText(/Spec/i);
    await user.click(screen.getByRole("button", { name: /Launch study/i }));

    expect(await screen.findByText(/Pick a study spec/i)).toBeInTheDocument();
  });

  it("surfaces an empty-state when no specs are registered", async () => {
    server.use(http.get(toMswPath(API_PATHS.configs), () => HttpResponse.json([])));
    renderWithProviders(<ConfigureStudyPage />);

    expect(await screen.findByText(/No study specs found/i)).toBeInTheDocument();
  });

  it("surfaces backend 422 errors inline", async () => {
    mockStudySpecsList();
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

    const select = await screen.findByLabelText(/Spec/i);
    await user.selectOptions(select, SPEC_NAME);
    await user.click(screen.getByRole("button", { name: /Launch study/i }));

    expect(await screen.findByText(/already running/i)).toBeInTheDocument();
  });
});
