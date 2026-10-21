import { useQuery } from "@tanstack/react-query";
import {
  Activity, BarChart3, Bell, BookOpen, Bot, BrainCircuit, Briefcase, CheckCircle2, Database, FileText, GitCompareArrows, LayoutDashboard,
  Library, Mail, Map as MapIcon, Menu, Moon, Network, Search, Send, Sun, Target, Users, Workflow, X,
} from "lucide-react";
import { ReactNode, useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api, getHealth } from "../lib/api";
import { cx, Dot } from "./ui";

export const NAV: { section: string; items: { to: string; label: string; icon: ReactNode; badge?: string }[] }[] = [
  { section: "Overview", items: [
    { to: "/", label: "Dashboard", icon: <LayoutDashboard className="h-4 w-4" /> },
    { to: "/pipeline", label: "Application pipeline", icon: <Workflow className="h-4 w-4" /> },
  ] },
  { section: "Recruiting", items: [
    { to: "/candidates", label: "Candidates", icon: <Users className="h-4 w-4" /> },
    { to: "/jobs", label: "Jobs", icon: <Briefcase className="h-4 w-4" /> },
    { to: "/matching", label: "Matching", icon: <Target className="h-4 w-4" /> },
    { to: "/reports", label: "Reports", icon: <FileText className="h-4 w-4" /> },
    { to: "/validation", label: "Human validation", icon: <CheckCircle2 className="h-4 w-4" />, badge: "pending" },
  ] },
  { section: "AI", items: [
    { to: "/assistant", label: "AI assistant", icon: <Bot className="h-4 w-4" /> },
    { to: "/search", label: "Semantic search", icon: <Search className="h-4 w-4" /> },
    { to: "/knowledge", label: "HR knowledge base", icon: <Library className="h-4 w-4" /> },
  ] },
  { section: "Automation", items: [
    { to: "/portal", label: "Candidate portal", icon: <Send className="h-4 w-4" /> },
    { to: "/outbox", label: "E-mails & meetings", icon: <Mail className="h-4 w-4" /> },
    { to: "/notifications", label: "Notifications & n8n", icon: <Bell className="h-4 w-4" />, badge: "unread" },
  ] },
  { section: "Models & results", items: [
    { to: "/models", label: "Base vs fine-tuned", icon: <GitCompareArrows className="h-4 w-4" /> },
    { to: "/evaluation", label: "Evaluation & MLflow", icon: <BarChart3 className="h-4 w-4" /> },
  ] },
  { section: "Under the hood", items: [
    { to: "/architecture", label: "Architecture (live)", icon: <MapIcon className="h-4 w-4" /> },
    { to: "/observe/vectors", label: "Vectors & RAG", icon: <Network className="h-4 w-4" /> },
    { to: "/observe/matching", label: "Matching internals", icon: <BrainCircuit className="h-4 w-4" /> },
    { to: "/observe/database", label: "Database", icon: <Database className="h-4 w-4" /> },
  ] },
  { section: "Help", items: [{ to: "/guide", label: "Run guide", icon: <BookOpen className="h-4 w-4" /> }] },
];

function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch { /* private window */ }
  }, [dark]);
  return [dark, () => setDark(!dark)];
}

function StatusBar() {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 30_000, retry: 0 });
  const c = health.data?.components || {};
  const pills: [string, boolean | undefined, string?][] = health.isError
    ? [["API", false, "not reachable: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"]]
    : [["API", health.data ? true : undefined], ["PostgreSQL", c.postgresql?.healthy, c.postgresql?.details], ["Qdrant", c.qdrant?.healthy, c.qdrant?.details]];
  if (c.ai_server) pills.push(["Kaggle AI", c.ai_server.healthy, c.ai_server.details]);
  if (c.n8n) pills.push(["n8n", c.n8n.healthy, c.n8n.details]);
  const provider = c.llm_provider?.configured;
  return (
    <div className="hidden items-center gap-1.5 md:flex">
      {pills.map(([name, ok, title]) => (
        <span key={name} title={title} className="chip surface-2"><Dot tone={ok === undefined ? "slate" : ok ? "emerald" : "rose"} pulse={ok === false} />{name}</span>
      ))}
      {provider && <span className={cx("chip", provider === "mock" ? "border-amber-400/40 bg-amber-500/10 text-amber-700 dark:text-amber-300" : "surface-2")} title={provider === "mock" ? "LLM_PROVIDER=mock: AI answers are fixed fakes. Start the Kaggle AI server (Run guide)." : "Model provider"}><Activity className="h-3 w-3" />LLM: {provider}</span>}
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const [dark, toggle] = useTheme();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  const unread = useQuery({ queryKey: ["unread"], queryFn: () => api.get("/notifications?limit=1"), refetchInterval: 20_000, retry: 0 });
  const pending = useQuery({ queryKey: ["pending-count"], queryFn: () => api.get("/analyses?status=pending_review&limit=200"), refetchInterval: 20_000, retry: 0 });
  const badges: Record<string, number> = { unread: unread.data?.unread || 0, pending: pending.data?.count || 0 };
  const current = NAV.flatMap((s) => s.items).find((i) => (i.to === "/" ? location.pathname === "/" : location.pathname.startsWith(i.to)));

  const sidebar = (
    <nav className="flex h-full flex-col">
      <div className="flex items-center gap-3 px-5 pb-5 pt-6">
        <img src="/favicon.svg" className="h-9 w-9 rounded-xl shadow-lg shadow-indigo-500/20" alt="" />
        <div>
          <div className="font-display text-[17px] font-semibold leading-none tracking-tight">TalentLens</div>
          <div className="muted mt-1 text-[10px] uppercase tracking-[0.16em]">AI recruitment</div>
        </div>
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto px-3 pb-6">
        {NAV.map((s) => (
          <div key={s.section}>
            <div className="muted px-3 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.16em]">{s.section}</div>
            <div className="space-y-0.5">
              {s.items.map((i) => (
                <NavLink key={i.to} to={i.to} end={i.to === "/"}
                  className={({ isActive }) => cx("group relative flex items-center gap-3 rounded-xl px-3 py-2 text-[13px] font-medium transition",
                    isActive ? "bg-gradient-to-r from-indigo-500/15 to-violet-500/5 text-indigo-700 dark:text-white" : "muted hover:bg-slate-900/[0.04] hover:text-[color:var(--text)] dark:hover:bg-white/[0.04]")}>
                  {({ isActive }) => (
                    <>
                      {isActive && <span className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-gradient-to-b from-indigo-400 to-cyan-400" />}
                      <span className={isActive ? "text-indigo-500 dark:text-indigo-300" : ""}>{i.icon}</span>
                      <span className="flex-1 truncate">{i.label}</span>
                      {i.badge && badges[i.badge] > 0 && <span className="rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 px-1.5 py-0.5 text-[10px] font-bold text-white">{badges[i.badge]}</span>}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="border-t hairline px-5 py-4 text-[11px]">
        <div className="muted">Decision support only — a recruiter decides (KB-RR-001).</div>
        <div className="mt-2 flex gap-3"><a className="text-indigo-500 hover:underline" href="/docs" target="_blank" rel="noreferrer">API docs</a><a className="text-indigo-500 hover:underline" href="http://localhost:5678" target="_blank" rel="noreferrer">n8n</a><a className="text-indigo-500 hover:underline" href="http://localhost:5000" target="_blank" rel="noreferrer">MLflow</a></div>
      </div>
    </nav>
  );

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 border-r hairline bg-white/60 backdrop-blur-xl dark:bg-ink-950/60 lg:block">{sidebar}</aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-slate-950/50" onClick={() => setOpen(false)} />
          <aside className="absolute left-0 top-0 h-full w-72 border-r hairline bg-[color:var(--bg)]">{sidebar}</aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b hairline bg-white/70 px-4 backdrop-blur-xl dark:bg-ink-950/70 lg:px-8">
          <button className="btn-ghost btn-sm lg:hidden" onClick={() => setOpen(!open)}>{open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}</button>
          <div className="min-w-0 flex-1 truncate text-sm font-medium"><span className="muted">TalentLens /</span> {current?.label || ""}</div>
          <StatusBar />
          <NavLink to="/notifications" className="btn-ghost btn-sm relative"><Bell className="h-4 w-4" />{badges.unread > 0 && <span className="absolute -right-1 -top-1 grid h-4 min-w-4 place-items-center rounded-full bg-rose-500 px-1 text-[9px] font-bold text-white">{badges.unread}</span>}</NavLink>
          <button className="btn-ghost btn-sm" onClick={toggle} title="Theme">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</button>
        </header>
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1400px] px-4 py-8 lg:px-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
