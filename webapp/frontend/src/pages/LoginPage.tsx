import { Navigate, useSearchParams } from "react-router-dom";
import { useMe } from "@/api/auth";
import { LoginForm } from "@/features/auth/LoginForm";
import { resolveFromParam } from "@/lib/routes";

export function LoginPage() {
  const { data: user, isLoading } = useMe();
  const [params] = useSearchParams();

  if (isLoading) return null;
  if (user) return <Navigate to={resolveFromParam(params)} replace />;

  return (
    <div className="flex min-h-screen items-center justify-center bg-muted/30 p-6">
      <LoginForm />
    </div>
  );
}
