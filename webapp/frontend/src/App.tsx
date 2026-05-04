import { Navigate, Route, Routes } from "react-router-dom";
import { useMe } from "@/api/auth";
import { AppShell } from "@/components/layout/AppShell";
import { RequireAdmin } from "@/components/auth/RequireAdmin";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { AdminPage } from "@/pages/AdminPage";
import { LoginPage } from "@/pages/LoginPage";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { RunsPage } from "@/pages/RunsPage";
import { ROUTES } from "@/lib/routes";

function ProtectedShell() {
  const { data: user } = useMe();
  if (!user) return null;
  return <AppShell user={user} />;
}

export function App() {
  return (
    <Routes>
      <Route path={ROUTES.login} element={<LoginPage />} />
      <Route
        element={
          <RequireAuth>
            <ProtectedShell />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to={ROUTES.runs} replace />} />
        <Route path={ROUTES.runs} element={<RunsPage />} />
        <Route
          path={ROUTES.admin}
          element={
            <RequireAdmin>
              <AdminPage />
            </RequireAdmin>
          }
        />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
