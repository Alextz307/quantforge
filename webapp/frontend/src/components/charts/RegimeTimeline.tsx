import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { Plot } from "@/components/charts/plot";
import type { RegimeSliceDTO } from "@/api/regime";

export interface RegimeTimelineProps {
  slices: readonly RegimeSliceDTO[];
  height?: number;
}

export function RegimeTimeline({ slices, height = 220 }: RegimeTimelineProps) {
  const tracesByLabel = useMemo(() => {
    const groups = new Map<string, RegimeSliceDTO[]>();
    for (const s of slices) {
      const list = groups.get(s.label);
      if (list) list.push(s);
      else groups.set(s.label, [s]);
    }
    return groups;
  }, [slices]);

  const plotData = useMemo<Data[]>(() => {
    const out: Data[] = [];
    tracesByLabel.forEach((items, label) => {
      const starts = items.map((s) => new Date(s.start).getTime());
      const ends = items.map((s) => new Date(s.end).getTime());
      const trace: Data & { base?: number[] } = {
        type: "bar",
        orientation: "h",
        name: label,
        x: starts.map((start, i) => (ends[i] ?? start) - start),
        y: items.map(() => label),
        base: starts,
        hovertemplate: items.map((s) => `${label}<br>${s.start} → ${s.end}<extra></extra>`),
      };
      out.push(trace);
    });
    return out;
  }, [tracesByLabel]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height,
      margin: { l: 100, r: 20, t: 20, b: 60 },
      barmode: "stack",
      xaxis: { type: "date", title: { text: "Time" } },
      yaxis: { title: { text: "Regime" }, automargin: true },
      showlegend: true,
      legend: { orientation: "h", x: 0, y: -0.25 },
    }),
    [height],
  );

  if (slices.length === 0) {
    return (
      <div
        data-testid="regime-timeline-empty"
        className="text-sm text-muted-foreground py-12 text-center"
      >
        No regime slices to render.
      </div>
    );
  }

  return (
    <div data-testid="regime-timeline" data-slice-count={slices.length}>
      <Plot
        data={plotData}
        layout={layout}
        config={{ displayModeBar: true, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
