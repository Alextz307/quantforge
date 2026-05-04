import type { ReactNode } from "react";
import { useMe } from "@/api/auth";
import { ROLE_ADMIN } from "@/api/users";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface RequireAdminProps {
  children: ReactNode;
}

export function RequireAdmin({ children }: RequireAdminProps) {
  const { data: user } = useMe();
  if (user && user.role !== ROLE_ADMIN) {
    return (
      <div className="mx-auto max-w-lg p-8">
        <Alert variant="destructive">
          <AlertTitle>Admin only</AlertTitle>
          <AlertDescription>You do not have access to this page.</AlertDescription>
        </Alert>
      </div>
    );
  }
  return <>{children}</>;
}
