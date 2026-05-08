import { Link } from "react-router-dom";
import { Beaker, PlayCircle, type LucideIcon } from "lucide-react";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ROUTES } from "@/lib/routes";

interface KindCardProps {
  to: string;
  icon: LucideIcon;
  title: string;
  description: string;
}

function KindCard({ to, icon: Icon, title, description }: KindCardProps) {
  return (
    <Link
      to={to}
      className="group block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-lg"
    >
      <Card className="h-full transition-colors group-hover:border-foreground/40">
        <CardHeader>
          <div className="flex items-center gap-3">
            <Icon className="h-6 w-6 text-muted-foreground" />
            <CardTitle>{title}</CardTitle>
          </div>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
      </Card>
    </Link>
  );
}

export function ConfigureLandingPage() {
  return (
    <div className="max-w-4xl space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Configure</h1>
        <p className="text-sm text-muted-foreground">
          Pick what to launch. Both flows share the same data + strategy block; tune adds the HPO
          knobs.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <KindCard
          to={ROUTES.configureRun}
          icon={PlayCircle}
          title="New run"
          description="Single experiment with walk-forward folds. Lands on /jobs once spawned."
        />
        <KindCard
          to={ROUTES.configureTune}
          icon={Beaker}
          title="New tune"
          description="Optuna study over the strategy's suggest_params space. Live-monitors trials as they land."
        />
      </div>
    </div>
  );
}
