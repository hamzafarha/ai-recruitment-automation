import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowRight, Bot, Briefcase, CheckCircle2, Circle, Cpu, Database, FileText, Hourglass, RefreshCw, Rocket, Sparkles, Users, Workflow } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";
import { ErrorState } from "../components/common";
import { Badge, Card, CardHeader, Empty, GradeBadge, KeyVals, Loading, PageHeader, Spinner, Stat } from "../components/ui";
import { api, errorText } from "../lib/api";
import { axisProps, PALETTE, tooltipStyle } from "../lib/chart";
import { ANALYSIS_STATUS, APP_STATUS, ago, GRADE_LABELS, GRADE_TONE, secs } from "../lib/format";
import { TONES } from "../components/ui";

export default function Dashboard() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const dash = useQuery({ queryKey: ["dashboard"], queryFn: () => api.get("/dashboard"), refetchInterval: 15_000 });
  const status = useQuery({ queryKey: ["status"], queryFn: () => api.get("/admin/status"), refetchInterval: 30_000 });
  const apps = useQuery({ queryKey: ["applications", "recent"], queryFn: () => api.get("/applications?limit=6"), refetchInterval: 10_000 });
  const live = useQuery({ queryKey: ["eval-pipeline"], queryFn: () => api.get("/evaluation/pipeline"), refetchInterval: 30_000 });
  const setup = useMutation({
    mutationFn: () => api.post("/admin/setup", { demo_data: true }),
    onSuccess: (r: any) => { toast.success(`Set up: ${r.knowledge_chunks ?? r.knowledge ?? "knowledge"} indexed`); client.invalidateQueries(); },
    onError: (e) => toast.error(errorText(e)),
  });
  const reindex = useMutation({ mutationFn: () => api.post("/admin/reindex"), onSuccess: () => { toast.success("Search entries rebuilt"); client.invalidateQueries(); }, onError: (e) => toast.error(errorText(e)) });

  if (dash.isLoading) return <Loading label="Loading the dashboard…" rows={6} />;
  if (dash.error) return <ErrorState error={dash.error} />;
  const d = dash.data;
  const s = status.data || {};
  const ai = s.ai_server || {};
  const modelsOk = s.llm_provider && s.llm_provider !== "mock" && (!ai.used || ai.reachable);
  const steps = [
    { ok: !!modelsOk, title: "AI models connected", text: modelsOk ? `Qwen on the Kaggle AI server${ai.adapter ? " + fine-tuned adapter" : ""}` : "Run notebooks/kaggle/model_server.ipynb on Kaggle, paste its 4 lines into .env, restart the API (Run guide § 5)" },
    { ok: !!s.knowledge_ready, title: "HR knowledge base indexed", text: "15 documents chunked and embedded in Qdrant (RAG)" },
    { ok: d.jobs > 0 && d.candidates > 0, title: "Jobs and candidates", text: "Load the demo (5 real job postings, 12 synthetic candidates) or add your own" },
  ];
  const grades = [3, 2, 1, 0].map((g) => ({ name: `${g} · ${GRADE_LABELS[g]}`, value: d.grades?.[String(g)] || 0, color: TONES[GRADE_TONE[g]].solid }));
  const appStatus = Object.entries(d.applications || {}).map(([k, v]) => ({ name: APP_STATUS[k]?.label || k, value: v as number, color: TONES[APP_STATUS[k]?.tone || "slate"].solid })).filter((x) => x.value > 0);
  const stepTimes = (live.data?.steps || []).filter((x: any) => x.mean_seconds).map((x: any) => ({ name: x.label, seconds: x.mean_seconds }));

  return (
    <div>
      <PageHeader eyebrow="AI-assisted CV screening" title="Recruiting, end to end" icon={<Sparkles className="h-7 w-7" />}
        subtitle="Every application goes through one workflow — the CV is read, structured by our fine-tuned Qwen, screened by a LangGraph agent with RAG over the HR rules, then a recruiter decides and the candidate is e-mailed (with a video meeting for interviews)."
        actions={<><Link to="/pipeline" className="btn-primary"><Workflow className="h-4 w-4" /> Submit a CV</Link><Link to="/guide" className="btn-ghost">Run guide</Link></>} />

      {steps.some((x) => !x.ok) && (
        <Card className="mb-6 overflow-hidden">
          <div className="grid gap-0 md:grid-cols-[1fr_auto]">
            <div className="p-5">
              <div className="mb-3 flex items-center gap-2 font-display font-semibold"><Rocket className="h-4 w-4 text-indigo-500" /> Getting started</div>
              <div className="grid gap-3 md:grid-cols-3">
                {steps.map((x, i) => (
                  <div key={i} className="flex gap-3">
                    {x.ok ? <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-500" /> : <Circle className="mt-0.5 h-5 w-5 shrink-0 text-slate-400" />}
                    <div><div className="text-sm font-semibold">{i + 1}. {x.title}</div><div className="muted mt-0.5 text-xs leading-relaxed">{x.text}</div></div>
                  </div>
                ))}
              </div>
            </div>
            <div className="flex flex-col justify-center gap-2 border-t hairline p-5 md:border-l md:border-t-0">
              <button className="btn-primary" disabled={setup.isPending} onClick={() => setup.mutate()}>{setup.isPending ? <Spinner /> : <Database className="h-4 w-4" />} Index the HR knowledge + load the demo</button>
              <button className="btn-ghost btn-sm" disabled={reindex.isPending} onClick={() => reindex.mutate()}><RefreshCw className="h-3.5 w-3.5" /> Rebuild search entries</button>
            </div>
          </div>
        </Card>
      )}

      <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          { label: "Candidates", value: d.candidates, hint: `${d.analyses} analyses`, icon: <Users className="h-4 w-4" />, tone: "indigo", to: "/candidates" },
          { label: "Open jobs", value: d.open_jobs, hint: `${d.jobs} jobs in total`, icon: <Briefcase className="h-4 w-4" />, tone: "cyan", to: "/jobs" },
          { label: "Waiting for a decision", value: d.pending_reviews, hint: "Human validation", icon: <Hourglass className="h-4 w-4" />, tone: "amber", to: "/validation" },
          { label: "Shortlisted / interview", value: d.shortlisted, hint: d.average_score ? `average score ${d.average_score}/100` : "—", icon: <CheckCircle2 className="h-4 w-4" />, tone: "emerald", to: "/reports" },
        ].map((x, i) => (
          <motion.div key={x.label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }} onClick={() => navigate(x.to)} className="cursor-pointer">
            <Stat label={x.label} value={x.value} hint={x.hint} icon={x.icon} tone={x.tone} className="card-hover" />
          </motion.div>
        ))}
      </div>

      <div className="mb-6 grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader title="Latest applications" subtitle="Live: each one moves through the 12 steps of the workflow" icon={<Workflow className="h-4 w-4" />} actions={<Link to="/pipeline" className="btn-ghost btn-sm">All <ArrowRight className="h-3.5 w-3.5" /></Link>} />
          <div className="p-3">
            {apps.data?.data?.length ? apps.data.data.map((a: any) => (
              <Link key={a.id} to={`/pipeline/${a.id}`} className="flex items-center gap-4 rounded-xl px-3 py-2.5 transition hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]">
                <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-indigo-500/20 to-cyan-500/20 text-xs font-semibold">{(a.candidate_name || a.filename || "?").slice(0, 2).toUpperCase()}</div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{a.candidate_name || a.filename}</div>
                  <div className="muted truncate text-xs">{a.applied_for} · {a.stage_label}</div>
                </div>
                <div className="hidden w-32 sm:block">
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-500/15"><div className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400" style={{ width: `${(a.steps_done / a.steps_total) * 100}%` }} /></div>
                  <div className="muted mt-1 text-[10px]">{a.steps_done}/{a.steps_total} steps</div>
                </div>
                {a.grade !== null && a.grade !== undefined && <GradeBadge grade={a.grade} />}
                <Badge tone={APP_STATUS[a.status]?.tone} dot>{APP_STATUS[a.status]?.label}</Badge>
              </Link>
            )) : <Empty icon={<Workflow className="h-6 w-6" />} title="No application yet" action={<Link to="/pipeline" className="btn-primary btn-sm">Submit a CV</Link>}>Submit a CV to watch it go through the pipeline.</Empty>}
          </div>
        </Card>
        <Card>
          <CardHeader title="Grades given" subtitle="Rubric KB-RR-002 (after recruiter adjustments)" icon={<FileText className="h-4 w-4" />} />
          <div className="h-56 px-2 pb-2">
            <ResponsiveContainer>
              <BarChart data={grades} margin={{ top: 16, right: 12, left: -18, bottom: 0 }}>
                <XAxis dataKey="name" {...axisProps} interval={0} tick={{ ...axisProps.tick, fontSize: 10 }} />
                <YAxis allowDecimals={false} {...axisProps} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>{grades.map((g) => <Cell key={g.name} fill={g.color} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="mb-6 grid gap-4 xl:grid-cols-3">
        <Card>
          <CardHeader title="Applications by status" icon={<Workflow className="h-4 w-4" />} />
          <div className="h-56">
            {appStatus.length ? (
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={appStatus} dataKey="value" nameKey="name" innerRadius={52} outerRadius={80} paddingAngle={3} stroke="none">{appStatus.map((x) => <Cell key={x.name} fill={x.color} />)}</Pie>
                  <Tooltip {...tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
            ) : <Empty title="No data yet" />}
          </div>
          <div className="flex flex-wrap gap-2 px-5 pb-5">{appStatus.map((x) => <span key={x.name} className="chip"><span className="h-2 w-2 rounded-full" style={{ background: x.color }} />{x.name} · {x.value}</span>)}</div>
        </Card>
        <Card>
          <CardHeader title="Mean time per step" subtitle="Live pipeline, done steps (Kaggle GPU for the AI steps)" icon={<Hourglass className="h-4 w-4" />} actions={<Link to="/evaluation?tab=pipeline" className="btn-ghost btn-sm">Details</Link>} />
          <div className="h-64 px-2 pb-3">
            {stepTimes.length ? (
              <ResponsiveContainer>
                <BarChart data={stepTimes} layout="vertical" margin={{ top: 8, right: 16, left: 30, bottom: 0 }}>
                  <XAxis type="number" {...axisProps} />
                  <YAxis type="category" dataKey="name" width={110} {...axisProps} tick={{ ...axisProps.tick, fontSize: 10 }} />
                  <Tooltip {...tooltipStyle} formatter={(v: any) => secs(Number(v))} />
                  <Bar dataKey="seconds" radius={[0, 6, 6, 0]} fill={PALETTE[0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <Empty title="No step timed yet" />}
          </div>
        </Card>
        <Card>
          <CardHeader title="System" subtitle="Models and stores the app uses now" icon={<Cpu className="h-4 w-4" />} />
          <div className="space-y-4 p-5">
            <KeyVals items={[
              ["LLM", s.llm_model ? String(s.llm_model).split("/").pop() : s.llm_provider],
              ["QLoRA adapter", ai.adapter ? <Badge key="a" tone="indigo">loaded</Badge> : ai.used ? "none" : "—"],
              ["Embeddings", String(s.embedding_model || "—").split("/").pop()],
              ["Reranker", String(s.reranking_model || "—").split("/").pop()],
              ["Kaggle AI server", ai.used ? (ai.reachable ? <Badge key="k" tone="emerald" dot>reachable</Badge> : <Badge key="k" tone="rose" dot>not reachable</Badge>) : "not used"],
              ["Qdrant points", Object.values(s.collections || {}).reduce((a: number, b: any) => a + Number(b || 0), 0)],
            ]} />
            <div className="flex flex-wrap gap-2">
              <Link to="/assistant" className="btn-ghost btn-sm"><Bot className="h-3.5 w-3.5" /> Ask the assistant</Link>
              <Link to="/architecture" className="btn-ghost btn-sm">Architecture</Link>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader title="Recent analyses" icon={<FileText className="h-4 w-4" />} actions={<Link to="/reports" className="btn-ghost btn-sm">Reports</Link>} />
          <div className="p-3">
            {(d.recent_analyses || []).length ? d.recent_analyses.map((a: any) => (
              <Link key={a.id} to={`/reports?id=${a.id}`} className="flex items-center gap-3 rounded-xl px-3 py-2 text-sm hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]">
                <div className="min-w-0 flex-1 truncate"><b>{a.candidate_name}</b> <span className="muted">· {a.job_title}</span></div>
                <GradeBadge grade={a.final_grade ?? a.grade} />
                <Badge tone={ANALYSIS_STATUS[a.status]?.tone}>{ANALYSIS_STATUS[a.status]?.label}</Badge>
              </Link>
            )) : <Empty title="No analysis yet" />}
          </div>
        </Card>
        <Card>
          <CardHeader title="Recent notifications" icon={<Workflow className="h-4 w-4" />} actions={<Link to="/notifications" className="btn-ghost btn-sm">All</Link>} />
          <div className="p-3">
            {(d.recent_notifications || []).map((n: any) => (
              <div key={n.id} className="flex items-start gap-3 rounded-xl px-3 py-2 text-sm">
                <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${n.read ? "bg-slate-400" : "bg-indigo-500"}`} />
                <div className="min-w-0 flex-1"><div className="truncate font-medium">{n.title}</div><div className="muted text-xs">{n.event} · {ago(n.created_at)}</div></div>
              </div>
            ))}
            {!(d.recent_notifications || []).length && <Empty title="Nothing yet" />}
          </div>
        </Card>
      </div>
    </div>
  );
}
