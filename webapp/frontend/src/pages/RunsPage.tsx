import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function RunsPage() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
        <CardDescription>Persisted experiment runs surface here.</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">
          The runs table and detail views are wired up in the next iteration. The backend already
          serves data at <code>/api/runs</code>.
        </p>
      </CardContent>
    </Card>
  );
}
