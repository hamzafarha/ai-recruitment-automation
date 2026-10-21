import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Plus, Workflow } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ErrorState, SubmitCv } from "../components/common";
import { STEP_ICONS } from "../components/domain/Pipeline";
import { Badge, Card, CardHeader, cx, Empty, GradeBadge, Loading, PageHeader, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { APP_STATUS, DECISIONS, when } from "../lib/format";

export default function Pipeline() {
  const navigate = useNavigate();
  const [filter, setFilter] = useState("all");
  const [showForm, setShowForm] = useState(false);
  const apps = useQuery({ queryKey: ["applications"], queryFn: () => api.get("/applications"), refetchInterval: 5000 });
  const catalogue = useQuery({ queryKey: ["steps"], queryFn: () => api.get("/applications/steps") });
  const data: any[] = apps.data?.data || [];
  const counts = data.reduce((acc: Record<string, number>, a) => ({ ...acc, [a.status]: (acc[a.status] || 0) + 1 }), {});
  const shown = data.filter((a) => filter === "all" || a.status === filter);
  return (
    <div>
      <PageHeader eyebrow="Diagram parts 1 → 9" title="Application pipeline" icon={<Workflow className="h-7 w-7" />}
        subtitle="One workflow per application, from the CV to the final decision: intake → document processing → candidate JSON by the fine-tuned Qwen → storage and search index → acknowledgement e-mail → screening agent (LangGraph + RAG) → evidence-backed report → recruiters notified → human validation → final decision → answer e-mailed (with a video meeting for interviews) → closed."
        actions={<button className="btn-primary" onClick={() => setShowForm(!showForm)}><Plus className="h-4 w-4" /> Submit a CV</button>} />

      {showForm && (
        <Card className="mb-6">
          <CardHeader title="Submit a CV to the pipeline" subtitle="The file is read once, then deleted (personal data, KB-HR-004)." icon={<Plus className="h-4 w-4" />} />
          <div className="p-5"><SubmitCv onSubmitted={(id) => navigate(`/pipeline/${id}`)} /></div>
        </Card>
      )}

      <Tabs value={filter} onChange={setFilter} tabs={[{ id: "all", label: "All", count: data.length }, ...Object.entries(APP_STATUS).map(([k, v]) => ({ id: k, label: v.label, count: counts[k] || 0 }))]} />

      {apps.isLoading ? <Loading rows={5} /> : apps.error ? <ErrorState error={apps.error} /> : !shown.length ? (
        <Card><Empty icon={<Workflow className="h-6 w-6" />} title="No application here" action={<button className="btn-primary btn-sm" onClick={() => setShowForm(true)}>Submit a CV</button>}>Submit a CV here, from Candidates, or from the Candidate portal.</Empty></Card>
      ) : (
        <div className="space-y-3">
          {shown.map((a) => (
            <Link key={a.id} to={`/pipeline/${a.id}`} className="card card-hover block p-4">
              <div className="flex flex-wrap items-center gap-4">
                <div className="min-w-[200px] flex-1">
                  <div className="flex items-center gap-2 font-semibold">{a.candidate_name || a.filename}<Badge tone={APP_STATUS[a.status]?.tone} dot>{APP_STATUS[a.status]?.label}</Badge></div>
                  <div className="muted mt-1 text-xs">{a.applied_for} · {a.source} · run by {a.orchestrator} · {when(a.created_at)}</div>
                </div>
                <div className="flex items-center gap-1">
                  {(catalogue.data?.data || []).map((s: any, i: number) => {
                    const done = i < a.steps_done;
                    const here = a.stage === s.step && a.status !== "closed";
                    return (
                      <span key={s.step} title={s.label} className={cx("grid h-7 w-7 place-items-center rounded-lg border text-[10px] transition [&>svg]:h-3.5 [&>svg]:w-3.5",
                        done ? "border-emerald-400/50 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300"
                          : here ? (a.status === "failed" ? "border-rose-400 bg-rose-500/10 text-rose-500" : "border-amber-400 bg-amber-500/10 text-amber-500 animate-pulse")
                            : "hairline text-slate-400")}>
                        {STEP_ICONS[s.step]}
                      </span>
                    );
                  })}
                </div>
                <div className="flex items-center gap-2">
                  {a.grade !== null && a.grade !== undefined && <GradeBadge grade={a.grade} />}
                  {a.decision && <Badge tone={DECISIONS[a.decision]?.tone}>{DECISIONS[a.decision]?.label}</Badge>}
                  <ChevronRight className="muted h-4 w-4" />
                </div>
              </div>
              {a.status === "failed" && a.error && <div className="mt-2 truncate text-xs text-rose-600 dark:text-rose-300">Stopped at «{a.stage_label}»: {a.error}</div>}
            </Link>
          ))}
        </div>
      )}

      <Card className="mt-8">
        <CardHeader title="The 12 steps and what each one runs" subtitle="Fine-tuning is not a step: it ran once on Kaggle (QLoRA, 873 CVs); the extraction step uses the adapter it produced." icon={<Workflow className="h-4 w-4" />} />
        <div className="grid gap-3 p-5 md:grid-cols-2 xl:grid-cols-3">
          {(catalogue.data?.data || []).map((s: any, i: number) => (
            <div key={s.step} className="flex gap-3 rounded-xl border hairline p-3">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-indigo-500/10 text-indigo-500">{STEP_ICONS[s.step]}</span>
              <div><div className="text-sm font-semibold">{i + 1}. {s.label} <span className="muted text-xs font-normal">· part {s.part}</span></div><div className="muted mt-0.5 text-xs leading-relaxed">{s.what}</div></div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
