import { NavLink } from "react-router-dom";
import {
  BarChart3,
  Beaker,
  LayoutDashboard,
  ListChecks,
  PlayCircle,
  Shield,
  Target,
  Workflow,
} from "lucide-react";
import { ROLE_ADMIN, type UserPublic } from "@/api/users";
import { cn } from "@/lib/cn";
import { ROUTES } from "@/lib/routes";

interface SidebarProps {
  user: UserPublic;
}

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  adminOnly?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: ROUTES.configure, label: "Configure", icon: PlayCircle },
  { to: ROUTES.jobs, label: "Jobs", icon: Workflow },
  { to: ROUTES.runs, label: "Runs", icon: LayoutDashboard },
  { to: ROUTES.comparisons, label: "Comparisons", icon: BarChart3 },
  { to: ROUTES.holdout, label: "Holdout", icon: Target },
  { to: ROUTES.studies, label: "Studies", icon: ListChecks },
  { to: ROUTES.hpo, label: "HPO", icon: Beaker },
  { to: ROUTES.admin, label: "Admin", icon: Shield, adminOnly: true },
];

export function Sidebar({ user }: SidebarProps) {
  const items = NAV_ITEMS.filter(
    (item) => !item.adminOnly || user.role === ROLE_ADMIN,
  );
  return (
    <aside className="flex w-64 flex-col border-r bg-card">
      <div className="flex h-16 items-center border-b px-6">
        <span className="text-lg font-semibold">QuantForge</span>
      </div>
      <nav className="flex-1 space-y-1 p-4">
        {items.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
