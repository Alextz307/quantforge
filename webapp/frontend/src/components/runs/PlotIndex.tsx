import { plotDownloadUrl } from "@/api/runs";

export interface PlotIndexProps {
  experimentId: string;
  plots: readonly string[];
}

export function PlotIndex({ experimentId, plots }: PlotIndexProps) {
  if (plots.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No static figures produced for this run.</p>
    );
  }
  return (
    <ul className="flex flex-col gap-2" data-testid="plot-index">
      {plots.map((name) => (
        <li key={name}>
          <a
            href={plotDownloadUrl(experimentId, name)}
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
