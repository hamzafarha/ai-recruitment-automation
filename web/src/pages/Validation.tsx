import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, HelpCircle, TrendingDown, UserCheck, Workflow } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ErrorState } from "../components/common";
import { DecisionForm } from "../components/domain/DecisionForm";
import { ReportView } from "../components/domain/Report";
import { Badge, Callout, Card, CardHeader, Collapsible, cx, Empty, GradeBadge, Loading, PageHeader, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { ANALYSIS_STATUS, METHOD_LABEL, REQ_STATUS, when } from "../lib/format";

export default function Validation() {
  const [params, setParams] = useSearchParams();
  const [tab, setTab] = useState<"waiting" | "decided">("waiting");
  const pending = useQuery({ queryKey: ["analyses", "pending"], queryFn: () => api.get("/analyses?status=pending_review").then((r) => r.data as any[]), refetchInterval: 15_000 });
  const all = useQuery({ queryKey: ["analyses"], queryFn: () => api.get("/analyses").then((r) => r.data as any[]), enabled: tab === "decided" });
  const id = params.get("id") || pending.data?.[0]?.id || "";
  const analysis = useQuery({ queryKey: ["analysis", id], queryFn: () => api.get(`/analyses/${id}`), enabled: !!id && tab === "waiting" });
  const a = analysis.data;
  const report = a?.report || {};
  const must = (report.requirements || []).filter((r: any) => r.kind === "must");
  return (
    <div>
      <PageHeader eyebrow="Diagram part 9 · KB-RR-001" title="Human validation" icon={<UserCheck className="h-7 w-7" />}
        subtitle="The AI proposes, a recruiter decides. Every analysis waits here until a recruiter schedules an interview, shortlists, holds or rejects. The decision resumes the application's workflow: the candidate is e-mailed — with a Google Meet / Jitsi video meeting and a calendar invite when an interview is scheduled." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "waiting", label: "Waiting for a decision", count: pending.data?.length }, { id: "decided", label: "Decisions made" }]} />
      {tab === "waiting" && (pending.isLoading ? <Loading rows={5} /> : pending.error ? <ErrorState error={pending.error} /> : !pending.data?.length ? (
        <Card><Empty icon={<CheckCircle2 className="h-6 w-6" />} title="Nothing waits for a decision">New applications arrive through the Application pipeline; analyses can also be made in Matching.</Empty></Card>
      ) : (
        <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
          <Card className="max-h-[80vh] overflow-y-auto p-2">
            {pending.data.map((r) => (
              <button key={r.id} onClick={() => setParams({ id: r.id })} className={cx("w-full rounded-xl px-3 py-2.5 text-left transition", r.id === id ? "bg-indigo-500/10 ring-1 ring-indigo-400/40" : "hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]")}>
                <div className="flex items-center justify-between gap-2"><span className="truncate text-sm font-semibold">{r.candidate_name}</span><GradeBadge grade={r.grade} /></div>
                <div className="muted mt-0.5 truncate text-xs">{r.job_title} · {r.score}/100</div>
                <div className="mt-1 flex gap-1"><Badge tone="slate">{METHOD_LABEL[r.method]}</Badge>{r.application && <Badge tone="indigo">pipeline</Badge>}</div>
              </button>
            ))}
          </Card>
          <div className="space-y-5">
            {analysis.isLoading || !a ? <Card className="p-6"><Loading /></Card> : (
              <>
                <Card className="p-6">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div><div className="font-display text-xl font-semibold">{a.candidate_name} — {a.job_title}</div><div className="muted text-xs">{METHOD_LABEL[a.method]} · {when(a.created_at)}</div></div>
                    {a.application && <Link to={`/pipeline/${a.application.id}`} className="btn-ghost btn-sm"><Workflow className="h-3.5 w-3.5" /> Its workflow</Link>}
                  </div>
                  {a.application && <Callout tone="indigo" className="mb-4">From the application pipeline: your decision resumes this application's workflow (final decision → e-mail to the candidate → closed).</Callout>}
                  <div className="mb-4 flex flex-wrap items-center gap-3"><GradeBadge grade={report.grade} label={report.grade_label} /><span className="text-sm">score <b>{report.score}/100</b> · {report.action_label}</span>{["met", "partial", "missing"].map((s) => <Badge key={s} tone={REQ_STATUS[s].tone}>{REQ_STATUS[s].label}: {must.filter((r: any) => r.status === s).length}</Badge>)}<span className="muted text-xs">(must-haves)</span></div>
                  <p className="mb-4 text-sm leading-relaxed">{report.summary}</p>
                  <div className="mb-4 grid gap-3 md:grid-cols-2">
                    <div className="rounded-xl border hairline p-3"><div className="mb-1.5 flex items-center gap-2 text-sm font-semibold"><TrendingDown className="h-4 w-4 text-amber-500" /> Gaps to explore</div><ul className="space-y-1 text-[13px]">{(report.gaps || []).map((g: string, i: number) => <li key={i}>· {g}</li>)}</ul></div>
                    <div className="rounded-xl border hairline p-3"><div className="mb-1.5 flex items-center gap-2 text-sm font-semibold"><HelpCircle className="h-4 w-4 text-violet-500" /> Interview questions</div><ul className="space-y-1 text-[13px]">{(report.interview_questions || []).map((g: string, i: number) => <li key={i}>· {g}</li>)}</ul></div>
                  </div>
                  <Collapsible title="Full report (requirements, evidence, HR rules used)"><ReportView analysis={a} /></Collapsible>
                </Card>
                <Card>
                  <CardHeader title="Your decision" subtitle="Stored with your name; a rejection needs a written reason (internal)." icon={<UserCheck className="h-4 w-4" />} />
                  <div className="p-5"><DecisionForm key={a.id} analysis={a} onDone={() => setParams({})} /></div>
                </Card>
              </>
            )}
          </div>
        </div>
      ))}
      {tab === "decided" && (all.isLoading ? <Loading /> : (
        <Card className="overflow-x-auto p-3">
          <table className="table-x">
            <thead><tr><th>Candidate</th><th>Job</th><th>Decision</th><th>AI grade</th><th>Final</th><th>Reviewer</th><th>Interview</th><th>Notes</th><th>When</th></tr></thead>
            <tbody>{(all.data || []).filter((x) => x.status !== "pending_review").map((x) => (
              <tr key={x.id}>
                <td className="font-medium"><Link className="hover:underline" to={`/reports?id=${x.id}`}>{x.candidate_name}</Link></td><td>{x.job_title}</td>
                <td><Badge tone={ANALYSIS_STATUS[x.status]?.tone}>{ANALYSIS_STATUS[x.status]?.label}</Badge></td><td>{x.grade}</td><td>{x.final_grade ?? "—"}</td>
                <td>{x.last_review?.reviewer}</td><td>{x.last_review?.interview_at || "—"}</td><td className="max-w-[240px] text-xs">{x.last_review?.notes || ""}</td><td className="muted text-xs">{when(x.last_review?.created_at)}</td>
              </tr>
            ))}</tbody>
          </table>
        </Card>
      ))}
    </div>
  );
}
