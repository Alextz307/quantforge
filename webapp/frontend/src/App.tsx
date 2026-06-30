import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { useMe } from "@/api/auth";
import { AppShell } from "@/components/layout/AppShell";
import { RequireAdmin } from "@/components/auth/RequireAdmin";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { HomePage } from "@/pages/HomePage";
import { LoginPage } from "@/pages/LoginPage";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { ROUTES } from "@/lib/routes";

// List pages are lazy so their bundles load on navigation rather than weighing
// down the initial paint of the home/login routes.
const RunsPage = lazy(() => import("@/pages/RunsPage").then((m) => ({ default: m.RunsPage })));
const ComparisonsPage = lazy(() =>
  import("@/pages/ComparisonsPage").then((m) => ({ default: m.ComparisonsPage })),
);
const HoldoutPage = lazy(() =>
  import("@/pages/HoldoutPage").then((m) => ({ default: m.HoldoutPage })),
);
const HpoPage = lazy(() => import("@/pages/HpoPage").then((m) => ({ default: m.HpoPage })));
const StudiesPage = lazy(() =>
  import("@/pages/StudiesPage").then((m) => ({ default: m.StudiesPage })),
);
const DeploymentsPage = lazy(() =>
  import("@/pages/DeploymentsPage").then((m) => ({ default: m.DeploymentsPage })),
);
const JobsPage = lazy(() => import("@/pages/JobsPage").then((m) => ({ default: m.JobsPage })));
const AdminPage = lazy(() => import("@/pages/AdminPage").then((m) => ({ default: m.AdminPage })));

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
const ConfigureComparePage = lazy(() =>
  import("@/pages/ConfigureComparePage").then((m) => ({ default: m.ConfigureComparePage })),
);
const ConfigureHoldoutPage = lazy(() =>
  import("@/pages/ConfigureHoldoutPage").then((m) => ({ default: m.ConfigureHoldoutPage })),
);
const ConfigureStudyPage = lazy(() =>
  import("@/pages/ConfigureStudyPage").then((m) => ({ default: m.ConfigureStudyPage })),
);
const ConfigureUniversePage = lazy(() =>
  import("@/pages/ConfigureUniversePage").then((m) => ({ default: m.ConfigureUniversePage })),
);
const JobDetailPage = lazy(() =>
  import("@/pages/JobDetailPage").then((m) => ({ default: m.JobDetailPage })),
);
const DeploymentDetailPage = lazy(() =>
  import("@/pages/DeploymentDetailPage").then((m) => ({ default: m.DeploymentDetailPage })),
);

function ChartFallback() {
  return <p className="text-sm text-muted-foreground">Loading...</p>;
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
        <Route index element={<HomePage />} />
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
        <Route
          path={ROUTES.configureCompare}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigureComparePage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.configureHoldout}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigureHoldoutPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.configureStudy}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigureStudyPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.configureUniverse}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ConfigureUniversePage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.jobs}
          element={
            <Suspense fallback={<ChartFallback />}>
              <JobsPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.jobDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <JobDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.runs}
          element={
            <Suspense fallback={<ChartFallback />}>
              <RunsPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.runDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <RunDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.comparisons}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ComparisonsPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.comparisonDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <ComparisonDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.holdout}
          element={
            <Suspense fallback={<ChartFallback />}>
              <HoldoutPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.holdoutDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <HoldoutDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.studies}
          element={
            <Suspense fallback={<ChartFallback />}>
              <StudiesPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.studyDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <StudyDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.hpo}
          element={
            <Suspense fallback={<ChartFallback />}>
              <HpoPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.hpoDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <HpoDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.deployments}
          element={
            <Suspense fallback={<ChartFallback />}>
              <DeploymentsPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.deploymentDetail}
          element={
            <Suspense fallback={<ChartFallback />}>
              <DeploymentDetailPage />
            </Suspense>
          }
        />
        <Route
          path={ROUTES.admin}
          element={
            <RequireAdmin>
              <Suspense fallback={<ChartFallback />}>
                <AdminPage />
              </Suspense>
            </RequireAdmin>
          }
        />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
