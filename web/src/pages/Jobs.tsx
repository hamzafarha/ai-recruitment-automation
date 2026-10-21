import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Briefcase, Plus, Save, Sparkles, Target, Trash2, Wand2, X } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { ErrorState } from "../components/common";
import { Badge, Card, CardHeader, Drawer, Empty, Field, Loading, Markdown, PageHeader, Spinner, Tabs } from "../components/ui";
import { api, errorText } from "../lib/api";
import { when } from "../lib/format";

const ROLES = ["", "ml_engineer", "nlp_llm_engineer", "backend_developer", "devops_cloud", "data_analyst_scientist"];
const EMPTY = { title: "", company: "", location: "", employment_type: "", role: "", description: "", requirements: [{ requirement_text: "", requirement_type: "required", importance: "high", min_years_experience: null }] };

function RequirementsList({ requirements }: { requirements: any[] }) {
  return (
    <div className="space-y-1.5">
      {requirements.map((r, i) => (
        <div key={i} className="flex items-start gap-3 rounded-xl border hairline px-3 py-2 text-sm">
          <Badge tone={r.requirement_type === "required" ? "indigo" : "slate"}>{r.id || (r.requirement_type === "required" ? "must" : "nice")}</Badge>
          <span className="flex-1">{r.requirement_text}</span>
          <span className="muted text-xs">{r.importance}{r.min_years_experience ? ` · ${r.min_years_experience}+ y` : ""}{r.hard ? " · hard" : ""}</span>
        </div>
      ))}
    </div>
  );
}

function JobEditor({ draft, onSaved }: { draft: any; onSaved: () => void }) {
  const client = useQueryClient();
  const [job, setJob] = useState<any>(() => ({ ...EMPTY, ...draft, requirements: draft.requirements?.length ? draft.requirements : EMPTY.requirements }));
  const set = (k: string, v: any) => setJob((j: any) => ({ ...j, [k]: v }));
  const setReq = (i: number, k: string, v: any) => setJob((j: any) => ({ ...j, requirements: j.requirements.map((r: any, n: number) => (n === i ? { ...r, [k]: v } : r)) }));
  const save = useMutation({
    mutationFn: () => api.post("/jobs", {
      ...job, role: job.role || null, company: job.company || null, location: job.location || null, employment_type: job.employment_type || null,
      requirements: job.requirements.filter((r: any) => r.requirement_text.trim()).map((r: any) => ({ requirement_text: r.requirement_text.trim(), requirement_type: r.requirement_type, importance: r.importance, min_years_experience: r.min_years_experience ? Number(r.min_years_experience) : null, hard: !!r.hard })),
    }),
    onSuccess: (j: any) => { toast.success(`Job “${j.title}” saved and indexed`); client.invalidateQueries(); onSaved(); },
    onError: (e) => toast.error(errorText(e)),
  });
  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-3">
        <Field label="Title"><input className="input" value={job.title} onChange={(e) => set("title", e.target.value)} /></Field>
        <Field label="Company"><input className="input" value={job.company || ""} onChange={(e) => set("company", e.target.value)} /></Field>
        <Field label="Knowledge-base job profile" hint="Which HR job profile the RAG uses for accepted equivalences">
          <select className="input" value={job.role || ""} onChange={(e) => set("role", e.target.value)}>{ROLES.map((r) => <option key={r} value={r}>{r || "— none —"}</option>)}</select>
        </Field>
        <Field label="Location"><input className="input" value={job.location || ""} onChange={(e) => set("location", e.target.value)} /></Field>
        <Field label="Employment type"><input className="input" value={job.employment_type || ""} onChange={(e) => set("employment_type", e.target.value)} /></Field>
      </div>
      <Field label="Description"><textarea className="input min-h-[90px]" value={job.description} onChange={(e) => set("description", e.target.value)} /></Field>
      <div>
        <div className="label">Requirements (M = must-have, N = nice-to-have)</div>
        <div className="space-y-2">
          {job.requirements.map((r: any, i: number) => (
            <div key={i} className="grid gap-2 md:grid-cols-[1fr_140px_120px_100px_36px]">
              <input className="input" placeholder="e.g. 3+ years of Python in production" value={r.requirement_text} onChange={(e) => setReq(i, "requirement_text", e.target.value)} />
              <select className="input" value={r.requirement_type} onChange={(e) => setReq(i, "requirement_type", e.target.value)}><option value="required">must-have</option><option value="preferred">nice-to-have</option></select>
              <select className="input" value={r.importance} onChange={(e) => setReq(i, "importance", e.target.value)}><option>high</option><option>medium</option><option>low</option></select>
              <input className="input" type="number" min={0} placeholder="min y" value={r.min_years_experience ?? ""} onChange={(e) => setReq(i, "min_years_experience", e.target.value || null)} />
              <button className="btn-ghost btn-sm" onClick={() => setJob((j: any) => ({ ...j, requirements: j.requirements.filter((_: any, n: number) => n !== i) }))}><X className="h-3.5 w-3.5" /></button>
            </div>
          ))}
        </div>
        <button className="btn-ghost btn-sm mt-2" onClick={() => set("requirements", [...job.requirements, { ...EMPTY.requirements[0], importance: "medium" }])}><Plus className="h-3.5 w-3.5" /> Requirement</button>
      </div>
      <div className="flex justify-end"><button className="btn-primary" disabled={!job.title.trim() || save.isPending} onClick={() => save.mutate()}>{save.isPending ? <Spinner /> : <Save className="h-4 w-4" />} Save the job</button></div>
    </div>
  );
}

export default function Jobs() {
  const client = useQueryClient();
  const [tab, setTab] = useState<"all" | "ad" | "manual">("all");
  const [open, setOpen] = useState<string | null>(null);
  const [ad, setAd] = useState("");
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api.get("/jobs").then((r) => r.data as any[]) });
  const job = useQuery({ queryKey: ["job", open], queryFn: () => api.get(`/jobs/${open}`), enabled: !!open });
  const extract = useMutation({ mutationFn: () => api.post("/jobs/extract", { text: ad }), onError: (e) => toast.error(errorText(e)) });
  const status = useMutation({ mutationFn: (s: string) => api.patch(`/jobs/${open}`, { status: s }), onSuccess: () => client.invalidateQueries() });
  const del = useMutation({ mutationFn: () => api.del(`/jobs/${open}`), onSuccess: () => { toast.success("Job deleted"); setOpen(null); client.invalidateQueries(); } });
  return (
    <div>
      <PageHeader eyebrow="Diagram parts 1, 6" title="Jobs" icon={<Briefcase className="h-7 w-7" />} subtitle="Job descriptions and their structured requirements (M1.. must-haves, N1.. nice-to-haves): every candidate is graded against them, requirement by requirement."
        actions={<button className="btn-primary" onClick={() => setTab("ad")}><Wand2 className="h-4 w-4" /> New job from an advertisement</button>} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "all", label: "All jobs", count: jobs.data?.length }, { id: "ad", label: "From an advertisement (AI)" }, { id: "manual", label: "Manual" }]} />
      {tab === "all" && (jobs.isLoading ? <Loading rows={5} /> : jobs.error ? <ErrorState error={jobs.error} /> : jobs.data?.length ? (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {jobs.data.map((j) => (
            <Card key={j.id} hover onClick={() => setOpen(j.id)} className="p-5">
              <div className="flex items-start justify-between gap-3"><div className="font-display font-semibold">{j.title}</div><Badge tone={j.status === "open" ? "emerald" : "slate"} dot>{j.status}</Badge></div>
              <div className="muted mt-1 text-xs">{[j.company, j.location, j.employment_type].filter(Boolean).join(" · ") || "—"}</div>
              <div className="mt-3 flex flex-wrap gap-1.5"><Badge tone="indigo">{j.must_haves} must-haves</Badge><Badge tone="slate">{j.nice_to_haves} nice-to-haves</Badge>{j.role && <Badge tone="cyan">{j.role}</Badge>}<Badge tone="slate">{j.source}</Badge></div>
            </Card>
          ))}
        </div>
      ) : <Card><Empty icon={<Briefcase className="h-6 w-6" />} title="No job yet">Load the demo on the Dashboard, or create one.</Empty></Card>)}
      {tab === "ad" && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card>
            <CardHeader title="Paste the advertisement" subtitle="Qwen structures it: title and requirements. Review, then save." icon={<Sparkles className="h-4 w-4" />} />
            <div className="space-y-3 p-5">
              <textarea className="input min-h-[320px] font-mono text-xs" value={ad} onChange={(e) => setAd(e.target.value)} placeholder="Senior Python Developer — Acme, Tunis…" />
              <div className="flex justify-end"><button className="btn-primary" disabled={ad.trim().length < 80 || extract.isPending} onClick={() => extract.mutate()}>{extract.isPending ? <Spinner /> : <Wand2 className="h-4 w-4" />} Structure with Qwen</button></div>
            </div>
          </Card>
          <Card>
            <CardHeader title="Review and save" icon={<Save className="h-4 w-4" />} />
            <div className="p-5">{extract.data ? <JobEditor key={JSON.stringify(extract.data).length} draft={extract.data} onSaved={() => setTab("all")} /> : <Empty title="Nothing structured yet" />}</div>
          </Card>
        </div>
      )}
      {tab === "manual" && <Card className="p-5"><JobEditor draft={EMPTY} onSaved={() => setTab("all")} /></Card>}
      <Drawer open={!!open} onClose={() => setOpen(null)} title={job.data?.title || "Job"} subtitle={job.data ? `${job.data.company || ""} · created ${when(job.data.created_at)}` : ""}
        actions={job.data && <>
          <Link to={`/matching?job=${open}`} className="btn-primary btn-sm"><Target className="h-3.5 w-3.5" /> Rank candidates</Link>
          <button className="btn-ghost btn-sm" onClick={() => status.mutate(job.data.status === "open" ? "closed" : "open")}>{job.data.status === "open" ? "Close" : "Reopen"}</button>
          <button className="btn-danger btn-sm" onClick={() => confirm("Delete this job and its analyses?") && del.mutate()}><Trash2 className="h-3.5 w-3.5" /></button>
        </>}>
        {job.isLoading || !job.data ? <Loading /> : (
          <div className="space-y-5">
            <RequirementsList requirements={job.data.requirements || []} />
            {job.data.description && <div className="rounded-2xl border hairline p-4"><Markdown>{job.data.description}</Markdown></div>}
          </div>
        )}
      </Drawer>
    </div>
  );
}
