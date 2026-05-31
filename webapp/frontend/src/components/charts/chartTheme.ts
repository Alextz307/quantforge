import type { Layout } from "plotly.js";
import { useMemo } from "react";
import { useTheme } from "@/lib/theme";
import type { ResolvedTheme } from "@/lib/themeStorage";

interface ChartPalette {
  font: string;
  grid: string;
  zeroLine: string;
  axisLine: string;
}

// Hex values mirror the HSL CSS tokens in src/index.css; baked here because
// Plotly can't read CSS vars (and to avoid a layout-time getComputedStyle).
const PALETTES: Record<ResolvedTheme, ChartPalette> = {
  light: { font: "#0a0a0c", grid: "#e2e5ea", zeroLine: "#cbd0d7", axisLine: "#cbd0d7" },
  dark: { font: "#f8fafc", grid: "#2a2d35", zeroLine: "#3f434c", axisLine: "#3f434c" },
};

function buildThemedLayout(resolved: ResolvedTheme): Partial<Layout> {
  const p = PALETTES[resolved];
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: p.font },
    xaxis: { gridcolor: p.grid, zerolinecolor: p.zeroLine, linecolor: p.axisLine },
    yaxis: { gridcolor: p.grid, zerolinecolor: p.zeroLine, linecolor: p.axisLine },
    legend: { font: { color: p.font } },
  };
}

function mergeThemedLayout(base: Partial<Layout>, resolved: ResolvedTheme): Partial<Layout> {
  const themed = buildThemedLayout(resolved);
  return {
    ...themed,
    ...base,
    xaxis: { ...themed.xaxis, ...base.xaxis },
    yaxis: { ...themed.yaxis, ...base.yaxis },
    font: { ...themed.font, ...base.font },
    legend: { ...themed.legend, ...base.legend },
  };
}

export function useThemedLayout(base: Partial<Layout>): Partial<Layout> {
  const { resolvedTheme } = useTheme();
  return useMemo(() => mergeThemedLayout(base, resolvedTheme), [base, resolvedTheme]);
}
