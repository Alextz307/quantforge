import {
  studyConsolidatedPlotUrl,
  studyConsolidatedTableUrl,
  type StudyConsolidatedDTO,
} from "@/api/studies";
import { MetadataField } from "@/components/MetadataField";
import { PlotIndex } from "@/components/PlotIndex";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDateTime } from "@/lib/format";

interface ConsolidatedReportPanelProps {
  dto: StudyConsolidatedDTO;
  // The study's directory name - used to build artifact URLs. Distinct from
  // ``dto.study_name``, which is the logical spec name from the manifest and
  // does not necessarily match the route segment.
  studyDirName: string;
}

export function ConsolidatedReportPanel({ dto, studyDirName }: ConsolidatedReportPanelProps) {
  return (
    <div className="flex flex-col gap-4" data-testid="consolidated-report-panel">
      <Card>
        <CardHeader>
          <CardTitle>Consolidated report</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <MetadataField label="Publish label" value={dto.publish_label} />
          <MetadataField label="Generated" value={formatDateTime(dto.created_at)} />
          <MetadataField label="Strategies" value={dto.strategies.length} />
          <MetadataField label="Universes" value={dto.universes.length} />
          <MetadataField label="Legs with holdout" value={dto.n_legs_with_holdout} />
          <MetadataField label="Universes w/ pairwise" value={dto.n_universes_with_pairwise} />
          <MetadataField label="Incomplete legs" value={dto.incomplete_leg_ids.length} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tables</CardTitle>
        </CardHeader>
        <CardContent>
          <PlotIndex
            plots={dto.tables}
            urlForPlot={(name) => studyConsolidatedTableUrl(studyDirName, name)}
            emptyMessage="No consolidated tables produced."
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Plots</CardTitle>
        </CardHeader>
        <CardContent>
          <PlotIndex
            plots={dto.plots}
            urlForPlot={(name) => studyConsolidatedPlotUrl(studyDirName, name)}
            emptyMessage="No consolidated plots produced for this study."
          />
        </CardContent>
      </Card>
    </div>
  );
}
