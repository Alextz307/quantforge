import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { useMe } from "@/api/auth";
import {
  useCreateDeployment,
  useDeleteDeployment,
  useDeployments,
  usePrefetchDeployment,
  type DeploymentSummary,
} from "@/api/deployments";
import { useHoldoutEvals, type HoldoutEvalSummary } from "@/api/holdout";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { FilterableTablePage } from "@/components/FilterableTablePage";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { QueryRenderer } from "@/components/QueryRenderer";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import { formatDate, formatMetric } from "@/lib/format";
import { deploymentDetailPath, ROUTES } from "@/lib/routes";
import { sourceKindLabel, type SourceKind } from "@/lib/sourceKind";

export function DeploymentsPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const [showPicker, setShowPicker] = useState(false);
  const query = useDeployments({ allUsers: isAdmin && allUsers });
  const deleteDeployment = useDeleteDeployment();

  function onDelete(d: DeploymentSummary) {
    if (!window.confirm(`Delete deployment "${d.name}"? Its signal log will be removed.`)) return;
    deleteDeployment.mutate(d.id);
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Deployments</CardTitle>
        <Button
          size="sm"
          onClick={() => {
            setShowPicker((v) => !v);
          }}
          data-testid="deployments-new-cta"
        >
          {showPicker ? "Close" : "+ New Deployment"}
        </Button>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="deployments"
          testId="deployments-all-users-toggle"
        />
        {showPicker && (
          <NewDeploymentPicker
            allUsers={isAdmin && allUsers}
            onCreated={() => {
              setShowPicker(false);
            }}
          />
        )}
        {deleteDeployment.isError && (
          <Alert variant="destructive">
            <AlertDescription>{deleteDeployment.error.message}</AlertDescription>
          </Alert>
        )}
        <QueryRenderer query={query} errorTitle="Failed to load deployments">
          {(rows) => (
            <DeploymentsTable
              rows={rows}
              onDelete={onDelete}
              deleting={deleteDeployment.isPending}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface TableProps {
  rows: readonly DeploymentSummary[];
  onDelete: (d: DeploymentSummary) => void;
  deleting: boolean;
}

function DeploymentsTable({ rows, onDelete, deleting }: TableProps) {
  const prefetch = usePrefetchDeployment();
  const sorted = useMemo(
    () =>
      [...rows].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [rows],
  );

  return (
    <FilterableTablePage<DeploymentSummary, Record<string, never>>
      rows={sorted}
      filters={{}}
      applyFilters={(r) => r}
      filterControls={null}
      rowKey={(r) => r.id}
      rowName={(r) => r.name}
      rowHref={(r) => deploymentDetailPath(r.id)}
      rowOnHover={(r) => {
        prefetch(r.id);
      }}
      tableTestId="deployments-table"
      emptyMessage="No deployments yet. Use “+ New Deployment” to deploy a trained model."
      columns={[
        { header: "Ticker", cellClassName: "font-mono", render: (r) => r.ticker },
        { header: "Strategy", render: (r) => r.strategy_name },
        { header: "Interval", render: (r) => r.interval },
        {
          header: "Trained through",
          cellClassName: "font-mono text-xs",
          render: (r) => formatDate(r.train_end),
        },
        { header: "Owner", render: (r) => <LaunchedByCell username={r.owner_username} /> },
        {
          header: "",
          align: "right",
          render: (r) => (
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Delete ${r.name}`}
              disabled={deleting}
              onClick={() => {
                onDelete(r);
              }}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          ),
        },
      ]}
    />
  );
}

interface PickerProps {
  allUsers: boolean;
  onCreated: () => void;
}

function NewDeploymentPicker({ allUsers, onCreated }: PickerProps) {
  const holdoutsQuery = useHoldoutEvals({ allUsers });
  const create = useCreateDeployment();
  const navigate = useNavigate();

  function deploy(sourceKind: SourceKind, sourceId: string) {
    create.mutate(
      { source_kind: sourceKind, source_id: sourceId },
      {
        onSuccess: (detail) => {
          onCreated();
          navigate(deploymentDetailPath(detail.id));
        },
      },
    );
  }

  return (
    <Card data-testid="deploy-picker">
      <CardHeader>
        <CardTitle className="text-lg">Deploy a holdout-evaluated model</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <p className="text-sm text-muted-foreground">
          Only models with a holdout evaluation appear here, ranked by out-of-sample Sharpe (best
          first). To deploy a run that hasn&apos;t been holdout-evaluated, open it from the{" "}
          <Link to={ROUTES.runs} className="text-primary hover:underline">
            Runs
          </Link>{" "}
          page and use Deploy there.
        </p>
        {create.isError && (
          <Alert variant="destructive">
            <AlertDescription>{create.error.message}</AlertDescription>
          </Alert>
        )}
        <QueryRenderer query={holdoutsQuery} errorTitle="Failed to load holdout evaluations">
          {(holdouts) => (
            <PickerTable holdouts={holdouts} onDeploy={deploy} deploying={create.isPending} />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface Candidate {
  sourceKind: SourceKind;
  sourceId: string;
  name: string;
  sharpe: number | null;
}

interface PickerTableProps {
  holdouts: readonly HoldoutEvalSummary[];
  onDeploy: (sourceKind: SourceKind, sourceId: string) => void;
  deploying: boolean;
}

function beatsSharpe(candidate: number | null, current: number | null): boolean {
  if (candidate === null) return false;
  if (current === null) return true;
  return candidate > current;
}

function PickerTable({ holdouts, onDeploy, deploying }: PickerTableProps) {
  const candidates = useMemo<Candidate[]>(() => {
    const best = new Map<string, Candidate>();
    for (const h of holdouts) {
      const key = `${h.source_kind}:${h.source_id}`;
      const prev = best.get(key);
      if (prev === undefined || beatsSharpe(h.sharpe_ratio, prev.sharpe)) {
        best.set(key, {
          sourceKind: h.source_kind,
          sourceId: h.source_id,
          name: h.name,
          sharpe: h.sharpe_ratio,
        });
      }
    }
    return [...best.values()].sort((a, b) => {
      if (a.sharpe === null && b.sharpe === null) return 0;
      if (a.sharpe === null) return 1;
      if (b.sharpe === null) return -1;
      return b.sharpe - a.sharpe;
    });
  }, [holdouts]);

  if (candidates.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No holdout-evaluated models yet. Run a holdout eval to deploy one from here.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="deploy-picker-table">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 pr-4">Holdout eval</th>
            <th className="py-2 pr-4">Source</th>
            <th className="py-2 pr-4 text-right">Holdout Sharpe</th>
            <th className="py-2 text-right" />
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr
              key={`${c.sourceKind}:${c.sourceId}`}
              className={cn("border-b last:border-0", c.sharpe === null && "text-muted-foreground")}
            >
              <td className="py-2 pr-4">{c.name}</td>
              <td className="py-2 pr-4 font-mono text-xs">
                {sourceKindLabel(c.sourceKind)} · {c.sourceId}
              </td>
              <td className="py-2 pr-4 text-right font-mono">{formatMetric(c.sharpe)}</td>
              <td className="py-2 text-right">
                <Button
                  size="sm"
                  disabled={deploying}
                  onClick={() => {
                    onDeploy(c.sourceKind, c.sourceId);
                  }}
                  data-testid={`deploy-${c.sourceKind}-${c.sourceId}`}
                >
                  Deploy
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
