declare module "plotly.js-cartesian-dist-min" {
  // Only the surface we call directly is typed; the rest of the bundle is
  // consumed structurally by react-plotly.js's factory.
  const plotly: {
    Plots: { resize(root: HTMLElement | string): Promise<void> };
    [key: string]: unknown;
  };
  export default plotly;
}
