import { useCallback, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMe } from "@/api/auth";
import { useCreateDeployment } from "@/api/deployments";
import { useHoldoutEvals, usePrefetchHoldoutEval, type HoldoutEvalSummary } from "@/api/holdout";
import { AllUsersToggle } from "@/components/AllUsersToggle";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FilterDate } from "@/components/FilterDate";
import {
  FilterableTablePage,
  type SortOrder,
  type SortState,
} from "@/components/FilterableTablePage";
import { FilterSelect } from "@/components/FilterSelect";
import { LaunchedByCell } from "@/components/LaunchedByCell";
import { QueryRenderer } from "@/components/QueryRenderer";
import { ALL_OPTION, uniqSorted } from "@/lib/filters";
import { formatDateTime, formatMetric } from "@/lib/format";
import { deploymentDetailPath, holdoutDetailPath, ROUTES } from "@/lib/routes";
import { SOURCE_KINDS, sourceKindLabel, type SourceKind } from "@/lib/sourceKind";

type SourceKindFilter = SourceKind | typeof ALL_OPTION;
type HoldoutSortKey = "created_at" | "holdout_start" | "sharpe_ratio";

const DEFAULT_SORT: SortState<HoldoutSortKey> = { sortBy: "created_at", order: "desc" };

function isSourceKindFilter(value: string): value is SourceKindFilter {
  return value === ALL_OPTION || (SOURCE_KINDS as readonly string[]).includes(value);
}

interface HoldoutFilters {
  sourceKind: SourceKindFilter;
  since: string;
}

function applyFilters(
  rows: readonly HoldoutEvalSummary[],
  f: HoldoutFilters,
): readonly HoldoutEvalSummary[] {
  const sinceMs = f.since ? new Date(f.since).getTime() : null;
  return rows.filter((r) => {
    if (f.sourceKind !== ALL_OPTION && r.source_kind !== f.sourceKind) return false;
    if (sinceMs != null && new Date(r.created_at).getTime() < sinceMs) return false;
    return true;
  });
}

function sortRows(
  rows: readonly HoldoutEvalSummary[],
  state: SortState<HoldoutSortKey>,
): readonly HoldoutEvalSummary[] {
  // In-flight evals carry ``sharpe_ratio=null``; mirror HpoPage's policy of
  // sinking nulls to the bottom under DESC (the "best first" reading).
  const dir = state.order === "desc" ? -1 : 1;
  const copied = [...rows];
  copied.sort((a, b) => {
    if (state.sortBy === "sharpe_ratio") {
      const av = a.sharpe_ratio;
      const bv = b.sharpe_ratio;
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return (av - bv) * dir;
    }
    const tsField = state.sortBy === "holdout_start" ? "holdout_start" : "created_at";
    return (new Date(a[tsField]).getTime() - new Date(b[tsField]).getTime()) * dir;
  });
  return copied;
}

export function HoldoutPage() {
  const me = useMe();
  const isAdmin = me.data?.role === "admin";
  const [allUsers, setAllUsers] = useState(false);
  const query = useHoldoutEvals({ allUsers: isAdmin && allUsers });
  const [sourceKind, setSourceKind] = useState<SourceKindFilter>(ALL_OPTION);
  const [since, setSince] = useState<string>("");
  const [sortState, setSortState] = useState<SortState<HoldoutSortKey>>(DEFAULT_SORT);

  const onSortToggle = useCallback((col: HoldoutSortKey) => {
    setSortState((prev) => {
      const nextOrder: SortOrder = prev.sortBy === col && prev.order === "desc" ? "asc" : "desc";
      return { sortBy: col, order: nextOrder };
    });
  }, []);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Holdout evaluations</CardTitle>
        <Button asChild size="sm">
          <Link to={ROUTES.configureHoldout} data-testid="holdout-new-cta">
            New holdout eval
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <AllUsersToggle
          isAdmin={isAdmin}
          checked={allUsers}
          onChange={setAllUsers}
          artifactLabel="holdout evaluations"
          testId="holdout-all-users-toggle"
        />
        <QueryRenderer query={query} errorTitle="Failed to load holdout evaluations">
          {(rows) => (
            <HoldoutBody
              rows={rows}
              sourceKind={sourceKind}
              since={since}
              onSourceKind={setSourceKind}
              onSince={setSince}
              sortState={sortState}
              onSortToggle={onSortToggle}
            />
          )}
        </QueryRenderer>
      </CardContent>
    </Card>
  );
}

interface BodyProps {
  rows: readonly HoldoutEvalSummary[];
  sourceKind: SourceKindFilter;
  since: string;
  onSourceKind: (v: SourceKindFilter) => void;
  onSince: (v: string) => void;
  sortState: SortState<HoldoutSortKey>;
  onSortToggle: (col: HoldoutSortKey) => void;
}

function HoldoutBody({
  rows,
  sourceKind,
  since,
  onSourceKind,
  onSince,
  sortState,
  onSortToggle,
}: BodyProps) {
  const sourceKindOptions = useMemo<SourceKind[]>(
    () => uniqSorted(rows.map((r) => r.source_kind)) as SourceKind[],
    [rows],
  );
  const filters = useMemo<HoldoutFilters>(() => ({ sourceKind, since }), [sourceKind, since]);
  const sorted = useMemo(() => sortRows(rows, sortState), [rows, sortState]);
  const prefetchHoldoutEval = usePrefetchHoldoutEval();
  const create = useCreateDeployment();
  const navigate = useNavigate();

  function deploy(source: HoldoutEvalSummary) {
    create.mutate(
      { source_kind: source.source_kind, source_id: source.source_id },
      {
        onSuccess: (d) => {
          navigate(deploymentDetailPath(d.id));
        },
      },
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {create.isError && (
        <Alert variant="destructive">
          <AlertDescription>{create.error.message}</AlertDescription>
        </Alert>
      )}
      <FilterableTablePage<HoldoutEvalSummary, HoldoutFilters, HoldoutSortKey>
        rows={sorted}
        filters={filters}
        applyFilters={applyFilters}
        filterControls={
          <>
            <FilterSelect
              id="filter-source-kind"
              label="Source kind"
              value={sourceKind}
              onChange={(next) => {
                if (isSourceKindFilter(next)) onSourceKind(next);
              }}
              allLabel="All source kinds"
              options={sourceKindOptions}
              optionLabel={(v) => sourceKindLabel(v as SourceKind)}
            />
            <FilterDate id="filter-since" label="Since" value={since} onChange={onSince} />
          </>
        }
        rowKey={(r) => r.name}
        rowName={(r) => r.name}
        rowHref={(r) => holdoutDetailPath(r.name)}
        rowOnHover={(r) => {
          prefetchHoldoutEval(r.name);
        }}
        tableTestId="holdout-table"
        emptyMessage="No holdout evaluations match the current filters."
        sortState={sortState}
        onSortToggle={onSortToggle}
        columns={[
          { header: "Store", cellClassName: "font-mono", render: (r) => r.store },
          {
            header: "Source",
            cellClassName: "font-mono text-xs",
            render: (r) => `${sourceKindLabel(r.source_kind)} · ${r.source_id}`,
          },
          {
            header: "Sharpe",
            align: "right",
            cellClassName: "font-mono",
            render: (r) => formatMetric(r.sharpe_ratio),
            sortKey: "sharpe_ratio",
          },
          {
            header: "Holdout start",
            cellClassName: "font-mono text-xs",
            render: (r) => formatDateTime(r.holdout_start),
            sortKey: "holdout_start",
          },
          {
            header: "Created",
            cellClassName: "font-mono text-xs",
            render: (r) => formatDateTime(r.created_at),
            sortKey: "created_at",
          },
          {
            header: "Launched by",
            render: (r) => <LaunchedByCell username={r.launched_by_username} />,
          },
          {
            header: "",
            align: "right",
            render: (r) => (
              <Button
                size="sm"
                variant="outline"
                disabled={create.isPending}
                onClick={() => {
                  deploy(r);
                }}
                data-testid={`deploy-holdout-${r.name}`}
              >
                Deploy
              </Button>
            ),
          },
        ]}
      />
    </div>
  );
}
