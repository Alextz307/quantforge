export interface PlotIndexProps {
  plots: readonly string[];
  urlForPlot: (plotName: string) => string;
  emptyMessage?: string;
}

export function PlotIndex({
  plots,
  urlForPlot,
  emptyMessage = "No static figures produced.",
}: PlotIndexProps) {
  if (plots.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyMessage}</p>;
  }
  return (
    <ul className="flex flex-col gap-2" data-testid="plot-index">
      {plots.map((name) => (
        <li key={name}>
          <a
            href={urlForPlot(name)}
            download={name}
            className="text-sm font-mono text-primary hover:underline"
          >
            {name}
          </a>
        </li>
      ))}
    </ul>
  );
}
