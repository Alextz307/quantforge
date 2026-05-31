import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { buildFeatureImportanceBars } from "@/components/charts/featureImportanceBars";
import type { FeatureImportanceResponse } from "@/api/runs";
import { renderWithProviders } from "../../util/render";

const PERMUTATION_COUNT = 3;
const GAIN_COUNT = 2;

const BOTH_METHODS: FeatureImportanceResponse = {
  computable: true,
  entries: [
    { feature: "rsi_14", importance: 0.4, std: 0.05, n_folds: 3, method: "permutation" },
    { feature: "vol_20", importance: 0.1, std: 0.02, n_folds: 3, method: "permutation" },
    { feature: "macd", importance: -0.02, std: 0.03, n_folds: 3, method: "permutation" },
    { feature: "rsi_14", importance: 0.6, std: 0.0, n_folds: 3, method: "xgb_gain" },
    { feature: "vol_20", importance: 0.3, std: 0.0, n_folds: 3, method: "xgb_gain" },
  ],
  message: null,
};

const PERMUTATION_ONLY: FeatureImportanceResponse = {
  computable: true,
  entries: [
    { feature: "rsi_14", importance: 0.4, std: 0.05, n_folds: 3, method: "permutation" },
    { feature: "vol_20", importance: 0.1, std: 0.02, n_folds: 3, method: "permutation" },
  ],
  message: null,
};

const EMPTY_WITH_MESSAGE: FeatureImportanceResponse = {
  computable: false,
  entries: [],
  message: "Feature importance was not computed for this run.",
};

const PERMUTATION_NULL_GAIN_FINITE: FeatureImportanceResponse = {
  computable: true,
  entries: [
    { feature: "rsi_14", importance: null, std: null, n_folds: 3, method: "permutation" },
    { feature: "vol_20", importance: null, std: null, n_folds: 3, method: "permutation" },
    { feature: "rsi_14", importance: 0.6, std: 0.0, n_folds: 3, method: "xgb_gain" },
    { feature: "vol_20", importance: 0.3, std: 0.0, n_folds: 3, method: "xgb_gain" },
  ],
  message: null,
};

const SINGLE_FOLD_PERMUTATION: FeatureImportanceResponse["entries"] = [
  { feature: "rsi_14", importance: 0.4, std: 0.0, n_folds: 1, method: "permutation" },
  { feature: "vol_20", importance: 0.1, std: 0.0, n_folds: 1, method: "permutation" },
];

describe("buildFeatureImportanceBars", () => {
  it("pairs features with values in ascending order so the largest sits on top", () => {
    const bars = buildFeatureImportanceBars(BOTH_METHODS.entries, "permutation");

    expect(bars.features).toEqual(["macd", "vol_20", "rsi_14"]);
    expect(bars.values).toEqual([-0.02, 0.1, 0.4]);
    expect(bars.errors).toEqual([0.03, 0.02, 0.05]);
  });

  it("omits error bars for xgb_gain", () => {
    const bars = buildFeatureImportanceBars(BOTH_METHODS.entries, "xgb_gain");

    expect(bars.features).toEqual(["vol_20", "rsi_14"]);
    expect(bars.errors).toBeNull();
  });

  it("drops entries whose importance is null", () => {
    const withNull: FeatureImportanceResponse["entries"] = [
      { feature: "rsi_14", importance: 0.4, std: 0.05, n_folds: 3, method: "permutation" },
      { feature: "broken", importance: null, std: null, n_folds: 3, method: "permutation" },
    ];

    const bars = buildFeatureImportanceBars(withNull, "permutation");

    expect(bars.features).toEqual(["rsi_14"]);
  });

  it("omits error bars when every across-fold std is zero (single-fold run)", () => {
    const bars = buildFeatureImportanceBars(SINGLE_FOLD_PERMUTATION, "permutation");

    expect(bars.errors).toBeNull();
  });
});

describe("FeatureImportanceChart", () => {
  it("renders the permutation method by default with the feature count", () => {
    renderWithProviders(<FeatureImportanceChart response={BOTH_METHODS} />);
    const wrapper = screen.getByTestId("feature-importance");

    expect(wrapper.getAttribute("data-method")).toBe("permutation");
    expect(wrapper.getAttribute("data-feature-count")).toBe(String(PERMUTATION_COUNT));
  });

  it("switches to xgb_gain when the toggle is clicked", () => {
    renderWithProviders(<FeatureImportanceChart response={BOTH_METHODS} />);

    fireEvent.click(screen.getByTestId("feature-importance-method-xgb_gain"));

    const wrapper = screen.getByTestId("feature-importance");
    expect(wrapper.getAttribute("data-method")).toBe("xgb_gain");
    expect(wrapper.getAttribute("data-feature-count")).toBe(String(GAIN_COUNT));
  });

  it("shows a method-specific methodology caption that follows the toggle", () => {
    renderWithProviders(<FeatureImportanceChart response={BOTH_METHODS} />);

    expect(screen.getByTestId("feature-importance-method-note")).toHaveTextContent(/permutation/i);

    fireEvent.click(screen.getByTestId("feature-importance-method-xgb_gain"));

    expect(screen.getByTestId("feature-importance-method-note")).toHaveTextContent(/xgboost gain/i);
  });

  it("defaults to the method with finite bars when permutation is all-null", () => {
    renderWithProviders(<FeatureImportanceChart response={PERMUTATION_NULL_GAIN_FINITE} />);

    const wrapper = screen.getByTestId("feature-importance");
    expect(wrapper.getAttribute("data-method")).toBe("xgb_gain");
    expect(screen.queryByTestId("feature-importance-method-permutation")).toBeNull();
  });

  it("hides the method toggle when only one method is present", () => {
    renderWithProviders(<FeatureImportanceChart response={PERMUTATION_ONLY} />);

    expect(screen.queryByTestId("feature-importance-method-permutation")).toBeNull();
    expect(screen.queryByTestId("feature-importance-method-xgb_gain")).toBeNull();
    expect(screen.getByTestId("feature-importance")).toBeInTheDocument();
  });

  it("renders the response message when there are no entries", () => {
    renderWithProviders(<FeatureImportanceChart response={EMPTY_WITH_MESSAGE} />);

    expect(screen.getByTestId("feature-importance-empty")).toHaveTextContent(
      EMPTY_WITH_MESSAGE.message ?? "",
    );
  });
});
