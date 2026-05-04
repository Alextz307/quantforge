import type { ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useMe } from "@/api/auth";
import { FROM_QUERY_PARAM, ROUTES } from "@/lib/routes";

interface RequireAuthProps {
  children: ReactNode;
}

export function RequireAuth({ children }: RequireAuthProps) {
  const { data: user, isLoading } = useMe();
  const location = useLocation();

  if (isLoading) {
    return <FullscreenSpinner />;
  }
  if (!user) {
    const from = location.pathname + location.search;
    return (
      <Navigate to={`${ROUTES.login}?${FROM_QUERY_PARAM}=${encodeURIComponent(from)}`} replace />
    );
  }
  return <>{children}</>;
}

function FullscreenSpinner() {
  return (
    <div className="flex h-screen items-center justify-center">
      <div className="text-sm text-muted-foreground">Loading…</div>
    </div>
  );
}
