import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-cartesian-dist-min";
import { useEffect, useLayoutEffect, useRef, type ComponentProps } from "react";

const PlotlyComponent = createPlotlyComponent(Plotly);

export function Plot(props: ComponentProps<typeof PlotlyComponent>) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (wrapper === null || typeof ResizeObserver === "undefined") return;
    // react-plotly only refits on a WINDOW resize, so a chart whose container
    // width changes without one (a scrollbar appearing, a late mount into a
    // laid-out page) keeps a stale width and overflows. Refit on container size.
    const observer = new ResizeObserver(() => {
      if (graphRef.current !== null) void Plotly.Plots.resize(graphRef.current);
    });
    observer.observe(wrapper);
    return () => {
      observer.disconnect();
    };
  }, []);

  // Also refit after every commit: a sibling re-render can reflow the page and
  // change our width with no resize for the observer to catch. useLayoutEffect
  // runs pre-paint (no flash); resize is a DOM op, not state, so it can't loop.
  useLayoutEffect(() => {
    if (graphRef.current !== null) void Plotly.Plots.resize(graphRef.current);
  });

  return (
    <div ref={wrapperRef} style={{ width: "100%", height: "100%" }}>
      <PlotlyComponent
        {...props}
        onInitialized={(_figure, graphDiv) => {
          graphRef.current = graphDiv;
        }}
      />
    </div>
  );
}
