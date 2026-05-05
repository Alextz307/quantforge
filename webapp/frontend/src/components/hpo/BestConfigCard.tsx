import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface BestConfigCardProps {
  config: Record<string, unknown>;
}

export function BestConfigCard({ config }: BestConfigCardProps) {
  const isEmpty = Object.keys(config).length === 0;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Best config</CardTitle>
      </CardHeader>
      <CardContent>
        {isEmpty ? (
          <p className="text-sm text-muted-foreground">No best-config snapshot recorded.</p>
        ) : (
          <pre
            data-testid="best-config-json"
            className="max-h-[480px] overflow-auto rounded-md bg-muted p-4 text-xs font-mono"
          >
            {JSON.stringify(config, null, 2)}
          </pre>
        )}
      </CardContent>
    </Card>
  );
}
