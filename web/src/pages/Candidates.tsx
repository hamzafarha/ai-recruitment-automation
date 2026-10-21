import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, MapPin, Search, Sparkles, Trash2, UserPlus, Users } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { ErrorState } from "../components/common";
import { ProfileCompare, ProfileView } from "../components/domain/Profile";
import { ANALYSIS_STATUS, METHOD_LABEL, when } from "../lib/format";
import { Badge, Callout, Card, CodeBlock, Drawer, Dropzone, Empty, GradeBadge, KeyVals, Loading, PageHeader, Spinner, Tabs } from "../components/ui";
import { api, errorText } from "../lib/api";

function CandidateDrawer({ id, onClose }: { id: string | null; onClose: () => void }) {
  const client = useQueryClient();
  const [tab, setTab] = useState<"profile" | "cv" | "analyses" | "models">("profile");
  const c = useQuery({ queryKey: ["candidate", id], queryFn: () => api.get(`/candidates/${id}`), enabled: !!id });
  const cv = useQuery({ queryKey: ["cv-text", id], queryFn: () => api.get(`/candidates/${id}/cv-text`), enabled: !!id && tab === "cv" });
  const compare = useMutation({ mutationFn: () => api.post(`/candidates/${id}/compare-models`), onSuccess: () => { toast.success("Compared with the base model"); client.invalidateQueries({ queryKey: ["candidate", id] }); }, onError: (e) => toast.error(errorText(e)) });
  const del = useMutation({ mutationFn: () => api.del(`/candidates/${id}`), onSuccess: () => { toast.success("Candidate deleted with their analyses, e-mails and search entries"); client.invalidateQueries(); onClose(); }, onError: (e) => toast.error(errorText(e)) });
  const d = c.data;
  const ext = d?.extraction || {};
  return (
    <Drawer open={!!id} onClose={onClose} width="max-w-5xl" title={d?.full_name || "Candidate"} subtitle={d ? `${d.source} · ${d.cv_filename || "no file"} · added ${when(d.created_at)}` : ""}
      actions={<button className="btn-danger btn-sm" disabled={del.isPending} onClick={() => confirm("Delete this candidate, their analyses, applications and e-mails? (right to erasure, KB-HR-004)") && del.mutate()}><Trash2 className="h-3.5 w-3.5" /> Delete</button>}>
      {c.isLoading || !d ? <Loading /> : (
        <>
          <KeyVals cols={4} items={[["Model", ext.model || ext.origin || "—"], ["Adapter", ext.adapter ? <Badge tone="indigo">QLoRA</Badge> : "—"], ["Input", ext.input_type || "—"], ["Search index", d.indexed ? <Badge tone="emerald" dot>indexed</Badge> : <Badge tone="rose">not indexed</Badge>]]} />
          <div className="my-5"><Tabs value={tab} onChange={setTab} tabs={[{ id: "profile", label: "Structured CV" }, { id: "cv", label: "CV text" }, { id: "analyses", label: "Analyses", count: d.analyses?.length || 0 }, { id: "models", label: "Base vs fine-tuned" }]} /></div>
          {tab === "profile" && <ProfileView profile={d.profile} />}
          {tab === "cv" && (cv.isLoading ? <Loading /> : <CodeBlock title={d.cv_filename || "CV text"} code={cv.data?.cv_text || "(no CV text stored: demo candidates come from a profile)"} maxHeight="max-h-[600px]" />)}
          {tab === "analyses" && (d.analyses?.length ? (
            <div className="space-y-2">{d.analyses.map((a: any) => (
              <Link key={a.id} to={`/reports?id=${a.id}`} className="flex flex-wrap items-center gap-3 rounded-xl border hairline p-3 text-sm hover:border-indigo-400/50">
                <span className="flex-1 font-medium">{a.job_title}</span><Badge tone="slate">{METHOD_LABEL[a.method]}</Badge><GradeBadge grade={a.final_grade ?? a.grade} /><span className="muted">{a.score}/100</span><Badge tone={ANALYSIS_STATUS[a.status]?.tone}>{ANALYSIS_STATUS[a.status]?.label}</Badge>
              </Link>))}</div>
          ) : <Empty title="No analysis yet" action={<Link className="btn-primary btn-sm" to="/matching">Go to Matching</Link>} />)}
          {tab === "models" && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="muted text-sm">Runs the base Qwen2.5-7B (without our QLoRA adapter) on this CV text and compares with the stored profile (fine-tuned).</div>
                <button className="btn-primary btn-sm" disabled={compare.isPending || !d.has_cv_text} onClick={() => compare.mutate()}>{compare.isPending ? <Spinner /> : <Sparkles className="h-3.5 w-3.5" />} {ext.model_comparison ? "Compare again" : "Compare"}</button>
              </div>
              {!d.has_cv_text && <Callout tone="amber">No CV text for this candidate (demo profile): use Base vs fine-tuned with a CV file instead.</Callout>}
              {ext.model_comparison ? <ProfileCompare result={ext.model_comparison} /> : <Empty icon={<Sparkles className="h-6 w-6" />} title="Not compared yet" />}
            </div>
          )}
        </>
      )}
    </Drawer>
  );
}

export default function Candidates() {
  const client = useQueryClient();
  const [tab, setTab] = useState<"all" | "add" | "test">("all");
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const list = useQuery({ queryKey: ["candidates"], queryFn: () => api.get("/candidates").then((r) => r.data as any[]) });
  const upload = useMutation({
    mutationFn: () => { const f = new FormData(); f.append("file", file!); return api.upload("/candidates", f); },
    onSuccess: (c: any) => { toast.success(`${c.full_name || "Candidate"} added and indexed`); client.invalidateQueries(); setFile(null); setTab("all"); setOpen(c.id); },
    onError: (e) => toast.error(errorText(e)),
  });
  const test = useMutation({ mutationFn: () => { const f = new FormData(); f.append("file", testFile!); return api.upload("/candidates/extract", f); }, onError: (e) => toast.error(errorText(e)) });
  const rows = (list.data || []).filter((c) => !q || `${c.full_name} ${c.location} ${c.headline}`.toLowerCase().includes(q.toLowerCase()));
  return (
    <div>
      <PageHeader eyebrow="Diagram parts 1, 3, 8" title="Candidates" icon={<Users className="h-7 w-7" />}
        subtitle="Every candidate as a structured profile (extracted by Qwen2.5-7B + our QLoRA adapter), stored in PostgreSQL and indexed in Qdrant for search. New applications should go through the Application pipeline; this page adds a candidate directly."
        actions={<button className="btn-primary" onClick={() => setTab("add")}><UserPlus className="h-4 w-4" /> Add a candidate</button>} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "all", label: "All candidates", count: list.data?.length }, { id: "add", label: "Add a candidate" }, { id: "test", label: "Test the extraction only" }]} />
      {tab === "all" && (list.isLoading ? <Loading rows={6} /> : list.error ? <ErrorState error={list.error} /> : (
        <>
          <div className="relative mb-4 max-w-md"><Search className="muted absolute left-3 top-2.5 h-4 w-4" /><input className="input pl-9" placeholder="Search by name, location, position…" value={q} onChange={(e) => setQ(e.target.value)} /></div>
          {rows.length ? (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {rows.map((c) => (
                <Card key={c.id} hover onClick={() => setOpen(c.id)} className="p-4">
                  <div className="flex items-start gap-3">
                    <div className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-500 text-sm font-semibold text-white">{(c.full_name || "?").split(" ").map((w: string) => w[0]).slice(0, 2).join("")}</div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-semibold">{c.full_name || "Unnamed"}</div>
                      <div className="muted truncate text-xs">{c.headline || "—"}</div>
                      <div className="muted mt-1 flex items-center gap-1 text-xs"><MapPin className="h-3 w-3" />{c.location || "—"}</div>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5"><Badge tone="indigo">{c.skills_count} skills</Badge><Badge tone="cyan">{c.experience_count} positions</Badge><Badge tone="slate">{c.source}</Badge>{!c.indexed && <Badge tone="rose">not indexed</Badge>}</div>
                </Card>
              ))}
            </div>
          ) : <Card><Empty icon={<Users className="h-6 w-6" />} title="No candidate">Load the demo on the Dashboard, or submit CVs.</Empty></Card>}
        </>
      ))}
      {tab === "add" && (
        <Card className="p-5">
          <Callout tone="indigo" className="mb-4">Tip: the <Link className="font-semibold underline" to="/pipeline">Application pipeline</Link> runs the same extraction and then the screening agent, e-mails and decision. Here the candidate is only extracted, stored and indexed.</Callout>
          <Dropzone file={file} onFile={setFile} />
          <div className="mt-4 flex justify-end"><button className="btn-primary" disabled={!file || upload.isPending} onClick={() => upload.mutate()}>{upload.isPending ? <Spinner /> : <UserPlus className="h-4 w-4" />} Extract and store (about a minute on the Kaggle GPU)</button></div>
        </Card>
      )}
      {tab === "test" && (
        <Card className="p-5">
          <div className="muted mb-4 text-sm">The structured profile of a CV, nothing stored (DSO1). To compare with the base model, use <Link className="underline" to="/models">Base vs fine-tuned</Link>.</div>
          <Dropzone file={testFile} onFile={setTestFile} />
          <div className="mt-4 flex justify-end"><button className="btn-primary" disabled={!testFile || test.isPending} onClick={() => test.mutate()}>{test.isPending ? <Spinner /> : <FlaskConical className="h-4 w-4" />} Extract</button></div>
          {test.data && (
            <div className="mt-6 space-y-4">
              <KeyVals cols={4} items={[["Provider", test.data.llm_provider], ["Model", test.data.stats?.model], ["Input", test.data.stats?.input_type], ["Time", test.data.stats?.seconds ? `${test.data.stats.seconds} s` : "—"]]} />
              <ProfileView profile={test.data.profile} />
            </div>
          )}
        </Card>
      )}
      <CandidateDrawer id={open} onClose={() => setOpen(null)} />
    </div>
  );
}
