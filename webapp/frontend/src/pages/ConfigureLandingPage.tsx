import {
  Beaker,
  GitCompareArrows,
  Globe,
  LayoutGrid,
  PlayCircle,
  Radio,
  ShieldCheck,
} from "lucide-react";
import { NavCard } from "@/components/NavCard";
import { ROUTES } from "@/lib/routes";

export function ConfigureLandingPage() {
  return (
    <div className="max-w-4xl space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Configure</h1>
        <p className="text-sm text-muted-foreground">
          Pick what to launch. Run + tune build experiments from scratch; compare, holdout + deploy
          reuse completed artifacts.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <NavCard
          to={ROUTES.configureRun}
          icon={PlayCircle}
          title="New run"
          description="Single experiment with walk-forward folds. Lands on /jobs once spawned."
        />
        <NavCard
          to={ROUTES.configureTune}
          icon={Beaker}
          title="New tune"
          description="Optuna study over the strategy's suggest_params space. Live-monitors trials as they land."
        />
        <NavCard
          to={ROUTES.configureCompare}
          icon={GitCompareArrows}
          title="New comparison"
          description="Rank 2-8 completed runs head-to-head with paired Sharpe-differential bootstrap."
        />
        <NavCard
          to={ROUTES.configureHoldout}
          icon={ShieldCheck}
          title="New holdout eval"
          description="Refit on full dev, evaluate once on the reserved holdout - honest OOS metrics."
        />
        <NavCard
          to={`${ROUTES.deployments}?new=1`}
          icon={Radio}
          title="New deployment"
          description="Deploy a trained model for live daily signals - pick a holdout-evaluated source, ranked by OOS Sharpe."
        />
        <NavCard
          to={ROUTES.configureStudy}
          icon={LayoutGrid}
          title="New study"
          description="Sweep a spec across strategies and universes; live leg grid as legs land."
        />
        <NavCard
          to={ROUTES.configureUniverse}
          icon={Globe}
          title="Manage universes"
          description="Upload reusable universe specs that study legs reference by slug."
        />
      </div>
    </div>
  );
}
