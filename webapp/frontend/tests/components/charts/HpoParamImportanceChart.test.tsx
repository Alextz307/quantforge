import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HpoParamImportanceChart } from "@/components/charts/HpoParamImportanceChart";
import { buildHpoImportanceRows } from "@/components/charts/hpoImportanceRows";
import type { ParamImportanceResponse } from "@/api/hpo";
import { renderWithProviders } from "../../util/render";

const POPULATED: ParamImportanceResponse = {
  importance: { window: 0.7, k: 0.3 },
  message: null,
};

const EMPTY_WITH_MESSAGE: ParamImportanceResponse = {
  importance: {},
  message: "Importance available after at least 2 completed trials.",
};

const EMPTY_NO_MESSAGE: ParamImportanceResponse = {
  importance: {},
  message: null,
};

describe("buildHpoImportanceRows", () => {
  it("pairs each name with its own value in ascending order", () => {
    const rows = buildHpoImportanceRows({ window: 0.7, k: 0.3 });

    expect(rows).toEqual([
      { name: "k", value: 0.3 },
      { name: "window", value: 0.7 },
    ]);
  });
});

describe("HpoParamImportanceChart", () => {
  it("renders the chart wrapper with the param count", () => {
    renderWithProviders(<HpoParamImportanceChart response={POPULATED} />);
    const wrapper = screen.getByTestId("hpo-importance");
    expect(wrapper.getAttribute("data-param-count")).toBe(
      String(Object.keys(POPULATED.importance).length),
    );
  });

  it("renders the response message when importance is empty", () => {
    renderWithProviders(<HpoParamImportanceChart response={EMPTY_WITH_MESSAGE} />);
    expect(screen.getByTestId("hpo-importance-empty")).toHaveTextContent(
      EMPTY_WITH_MESSAGE.message ?? "",
    );
  });

  it("renders a default message when importance is empty and no message provided", () => {
    renderWithProviders(<HpoParamImportanceChart response={EMPTY_NO_MESSAGE} />);
    expect(screen.getByTestId("hpo-importance-empty")).toHaveTextContent(/No importance data yet/i);
  });
});
