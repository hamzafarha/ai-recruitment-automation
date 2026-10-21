import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Trash2, Workflow } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ErrorState } from "../components/common";
import { AgentJobTrace } from "../components/domain/Agent";
import { ReportView } from "../components/domain/Report";
import { Badge, Card, cx, Empty, GradeBadge, Loading, PageHeader, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { ANALYSIS_STATUS, METHOD_LABEL, when } from "../lib/format";

export default function Reports() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState("all");
  const [view, setView] = useState<"report" | "trace">("report");
  const list = useQuery({ queryKey: ["analyses"], queryFn: () => api.get("/analyses").then((r) => r.data as any[]) });
  const rows = (list.data || []).filter((a) => filter === "all" || a.status === filter);
  const id = params.get("id") || rows[0]?.id || "";
  const analysis = useQuery({ queryKey: ["analysis", id], queryFn: () => api.get(`/analyses/${id}`), enabled: !!id });
  const del = useMutation({ mutationFn: () => api.del(`/analyses/${id}`), onSuccess: () => { toast.success("Analysis deleted"); setParams({}); client.invalidateQueries(); } });
  const a = analysis.data;
  const trace = a?.report?.trace;
  return (
    <div>
      <PageHeader eyebrow="Diagram part 6" title="Evidence-backed reports" icon={<FileText className="h-7 w-7" />} subtitle="Every analysis with its grade, score, the CV evidence for each requirement, the HR rules used, gaps and interview questions. Export as PDF or Markdown." />
      <Tabs value={filter} onChange={setFilter} tabs={[{ id: "all", label: "All", count: list.data?.length }, ...Object.entries(ANALYSIS_STATUS).map(([k, v]) => ({ id: k, label: v.label, count: (list.data || []).filter((x) => x.status === k).length }))]} />
      {list.isLoading ? <Loading rows={6} /> : list.error ? <ErrorState error={list.error} /> : !rows.length ? <Card><Empty icon={<FileText className="h-6 w-6" />} title="No report">Analyses come from the pipeline (screening agent) or from Matching.</Empty></Card> : (
        <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
          <Card className="max-h-[78vh] overflow-y-auto p-2">
            {rows.map((r) => (
              <button key={r.id} onClick={() => setParams({ id: r.id })} className={cx("w-full rounded-xl px-3 py-2.5 text-left transition", r.id === id ? "bg-indigo-500/10 ring-1 ring-indigo-400/40" : "hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]")}>
                <div className="flex items-center justify-between gap-2"><span className="truncate text-sm font-semibold">{r.candidate_name}</span><GradeBadge grade={r.final_grade ?? r.grade} /></div>
                <div className="muted mt-0.5 truncate text-xs">{r.job_title}</div>
                <div className="mt-1.5 flex flex-wrap gap-1"><Badge tone={ANALYSIS_STATUS[r.status]?.tone}>{ANALYSIS_STATUS[r.status]?.label}</Badge><Badge tone="slate">{METHOD_LABEL[r.method]}</Badge><span className="muted text-[10px]">{when(r.created_at)}</span></div>
              </button>
            ))}
          </Card>
          <Card className="p-6">
            {analysis.isLoading || !a ? <Loading /> : (
              <>
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <div><div className="font-display text-xl font-semibold">{a.candidate_name} — {a.job_title}</div><div className="muted text-xs">{METHOD_LABEL[a.method]} · {when(a.created_at)}</div></div>
                  <div className="flex gap-2">
                    {a.application && <Link to={`/pipeline/${a.application.id}`} className="btn-ghost btn-sm"><Workflow className="h-3.5 w-3.5" /> Its application</Link>}
                    {a.status === "pending_review" && <Link to={`/validation?id=${a.id}`} className="btn-primary btn-sm">Decide</Link>}
                    <button className="btn-danger btn-sm" onClick={() => confirm("Delete this analysis?") && del.mutate()}><Trash2 className="h-3.5 w-3.5" /></button>
                  </div>
                </div>
                {trace?.agent && <Tabs value={view} onChange={setView} tabs={[{ id: "report", label: "Report" }, { id: "trace", label: "Screening agent trace" }]} />}
                {view === "trace" && trace?.agent ? (
                  <AgentJobTrace job={{ agent: trace.agent, prompt: trace.prompt, prompt_chars: trace.prompt_chars, raw_answer: trace.raw_answer, llm_seconds: trace.llm_seconds, prompt_tokens: trace.prompt_tokens, output_tokens: trace.output_tokens, checks: trace.checks, error: trace.error }} />
                ) : <ReportView analysis={a} />}
                {!!a.reviews?.length && (
                  <div className="mt-6 space-y-2">
                    <div className="text-sm font-semibold">Recruiter decisions</div>
                    {a.reviews.map((r: any) => <div key={r.id} className="flex flex-wrap items-center gap-2 rounded-xl border hairline p-3 text-sm"><Badge tone="violet">{r.decision}</Badge><span>by {r.reviewer}</span><span className="muted">{when(r.created_at)}</span>{r.interview_at && <Badge tone="violet">{r.interview_at}</Badge>}{r.notes && <span className="muted italic">“{r.notes}”</span>}</div>)}
                  </div>
                )}
              </>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
