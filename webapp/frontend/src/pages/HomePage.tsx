import {
  BarChart3,
  Beaker,
  LayoutDashboard,
  ListChecks,
  PlayCircle,
  Target,
  Workflow,
} from "lucide-react";
import { NavCard } from "@/components/NavCard";
import { ROUTES } from "@/lib/routes";

export function HomePage() {
  return (
    <div className="max-w-5xl space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">QuantForge</h1>
        <p className="text-muted-foreground">
          Thesis-grade C++/Python quantitative trading framework with strict anti-leakage
          guarantees, walk-forward validation, and typed temporal contracts. The webapp is a single
          front door over the experiment lifecycle — configure runs, watch jobs land, drill through
          artifacts.
        </p>
      </header>
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <NavCard
          to={ROUTES.configure}
          icon={PlayCircle}
          title="Configure"
          description="Launch runs, tunes, comparisons, holdouts, and studies from form-driven configs."
        />
        <NavCard
          to={ROUTES.jobs}
          icon={Workflow}
          title="Jobs"
          description="Live monitor of running and recent jobs. WebSocket log tail per job."
        />
        <NavCard
          to={ROUTES.runs}
          icon={LayoutDashboard}
          title="Runs"
          description="Single-experiment walk-forward results — manifest, fold metrics, equity curves."
        />
        <NavCard
          to={ROUTES.studies}
          icon={ListChecks}
          title="Studies"
          description="Cross-strategy × cross-universe sweeps. Live leg grid; consolidated reports."
        />
        <NavCard
          to={ROUTES.hpo}
          icon={Beaker}
          title="HPO"
          description="Optuna study browser — live trial stream, convergence, hyperparameter importance."
        />
        <NavCard
          to={ROUTES.comparisons}
          icon={BarChart3}
          title="Comparisons"
          description="Rank completed runs head-to-head with paired Sharpe-differential bootstrap."
        />
        <NavCard
          to={ROUTES.holdout}
          icon={Target}
          title="Holdout"
          description="Refit on full dev, evaluate once on the reserved holdout — honest OOS metrics."
        />
      </section>
    </div>
  );
}
