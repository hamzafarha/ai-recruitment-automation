import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Loading } from "./components/ui";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Pipeline = lazy(() => import("./pages/Pipeline"));
const ApplicationDetail = lazy(() => import("./pages/ApplicationDetail"));
const Candidates = lazy(() => import("./pages/Candidates"));
const Jobs = lazy(() => import("./pages/Jobs"));
const Matching = lazy(() => import("./pages/Matching"));
const Reports = lazy(() => import("./pages/Reports"));
const Validation = lazy(() => import("./pages/Validation"));
const Assistant = lazy(() => import("./pages/Assistant"));
const Search = lazy(() => import("./pages/Search"));
const Knowledge = lazy(() => import("./pages/Knowledge"));
const Portal = lazy(() => import("./pages/Portal"));
const Outbox = lazy(() => import("./pages/Outbox"));
const Notifications = lazy(() => import("./pages/Notifications"));
const ModelComparison = lazy(() => import("./pages/ModelComparison"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Architecture = lazy(() => import("./pages/Architecture"));
const Vectors = lazy(() => import("./pages/Vectors"));
const MatchingInternals = lazy(() => import("./pages/MatchingInternals"));
const DatabasePage = lazy(() => import("./pages/Database"));
const Guide = lazy(() => import("./pages/Guide"));
const NotFound = lazy(() => import("./pages/NotFound"));

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<div className="p-2"><Loading /></div>}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/pipeline" element={<Pipeline />} />
          <Route path="/pipeline/:id" element={<ApplicationDetail />} />
          <Route path="/candidates" element={<Candidates />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/matching" element={<Matching />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/validation" element={<Validation />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/search" element={<Search />} />
          <Route path="/knowledge" element={<Knowledge />} />
          <Route path="/knowledge/:docId" element={<Knowledge />} />
          <Route path="/portal" element={<Portal />} />
          <Route path="/outbox" element={<Outbox />} />
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/observe/vectors" element={<Vectors />} />
          <Route path="/observe/matching" element={<MatchingInternals />} />
          <Route path="/observe/database" element={<DatabasePage />} />
          <Route path="/guide" element={<Guide />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
