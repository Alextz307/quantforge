import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useMe } from "@/api/auth";
import { AppShell } from "@/components/layout/AppShell";
import { RequireAdmin } from "@/components/auth/RequireAdmin";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { AdminPage } from "@/pages/AdminPage";
import { ComparisonsPage } from "@/pages/ComparisonsPage";
import { HoldoutPage } from "@/pages/HoldoutPage";
import { HpoPage } from "@/pages/HpoPage";
import { JobsPage } from "@/pages/JobsPage";
import { LoginPage } from "@/pages/LoginPage";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { RunsPage } from "@/pages/RunsPage";
import { StudiesPage } from "@/pages/StudiesPage";
import { ROUTES } from "@/lib/routes";

const RunDetailPage = lazy(() =>
  import("@/pages/RunDetailPage").then((m) => ({ default: m.RunDetailPage })),
);
const ComparisonDetailPage = lazy(() =>
  import("@/pages/ComparisonDetailPage").then((m) => ({ default: m.ComparisonDetailPage })),
);
const HoldoutDetailPage = lazy(() =>
  import("@/pages/HoldoutDetailPage").then((m) => ({ default: m.HoldoutDetailPage })),
);
const StudyDetailPage = lazy(() =>
  import("@/pages/StudyDetailPage").then((m) => ({ default: m.StudyDetailPage })),
);
const HpoDetailPage = lazy(() =>
  import("@/pages/HpoDetailPage").then((m) => ({ default: m.HpoDetailPage })),
);
const ConfigurePage = lazy(() =>
  import("@/pages/ConfigurePage").then((m) => ({ default: m.ConfigurePage })),
);
const ConfigureLandingPage = lazy(() =>
  import("@/pages/ConfigureLandingPage").then((m) => ({ default: m.ConfigureLandingPage })),
);
const ConfigureTunePage = lazy(() =>
  import("@/pages/ConfigureTunePage").then((m) => ({ default: m.ConfigureTunePage })),
);
const JobDetailPage = lazy(() =>
  import("@/pages/JobDetailPage").then((m) => ({ default: m.JobDetailPage })),
);

function ChartFallback() {
  return <p className="text-sm text-muted-foreground">Loading…</p>;
}

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
        <Route
          path={ROUTES.configure}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigureLandingPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.configureRun}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigurePage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.configureTune}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigureTunePage />
            </Suspense>
          }
        />
        <Route path={ROUTES.jobs} element={<JobsPage />} />
        <Route
          path={ROUTES.jobDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <JobDetailPage />
            </Suspense>
          }
        />
        <Route path={ROUTES.runs} element={<RunsPage />} />
        <Route
          path={ROUTES.runDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <RunDetailPage />
            </Suspense>
          }
        />
        <Route path={ROUTES.comparisons} element={<ComparisonsPage />} />
        <Route
          path={ROUTES.comparisonDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ComparisonDetailPage />
            </Suspense>
          }
        />
        <Route path={ROUTES.holdout} element={<HoldoutPage />} />
        <Route
          path={ROUTES.holdoutDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <HoldoutDetailPage />
            </Suspense>
          }
        />
        <Route path={ROUTES.studies} element={<StudiesPage />} />
        <Route
          path={ROUTES.studyDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <StudyDetailPage />
            </Suspense>
          }
        />
        <Route path={ROUTES.hpo} element={<HpoPage />} />
        <Route
          path={ROUTES.hpoDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <HpoDetailPage />
            </Suspense>
          }
        />
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
