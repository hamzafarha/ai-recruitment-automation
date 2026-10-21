import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Sparkles, Target, Zap } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ErrorState, JobSelect } from "../components/common";
import { ReportView } from "../components/domain/Report";
import { Badge, Card, CardHeader, Drawer, Empty, Field, GradeBadge, Loading, PageHeader, Progress, Spinner } from "../components/ui";
import { api, errorText } from "../lib/api";
import { ANALYSIS_STATUS, pct } from "../lib/format";

export default function Matching() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const jobId = params.get("job") || "";
  const [report, setReport] = useState<string | null>(null);
  const ranking = useQuery({ queryKey: ["ranking", jobId], queryFn: () => api.get(`/jobs/${jobId}/ranking`), enabled: !!jobId });
  const analysis = useQuery({ queryKey: ["analysis", report], queryFn: () => api.get(`/analyses/${report}`), enabled: !!report });
  const analyse = useMutation({
    mutationFn: ({ candidate, method }: { candidate: string; method: string }) => api.post("/analyses", { candidate_id: candidate, job_id: jobId, method }),
    onSuccess: (a: any) => { toast.success(`Analysis ready: grade ${a.grade}`); client.invalidateQueries({ queryKey: ["ranking", jobId] }); setReport(a.id); },
    onError: (e) => toast.error(errorText(e)),
  });
  const rows: any[] = ranking.data?.data || [];
  return (
    <div>
      <PageHeader eyebrow="DSO3 · diagram parts 5-6" title="Matching" icon={<Target className="h-7 w-7" />}
        subtitle="Every candidate graded for a job in seconds by the fast semantic matcher (BGE-M3 + BGE reranker + rubric KB-RR-002, no LLM). Then a full AI analysis (Qwen2.5-7B + HR rules retrieved by RAG, every claim checked by code) for the candidates worth it." />
      <Card className="mb-6 p-5">
        <Field label="Job"><JobSelect value={jobId} onChange={(v) => setParams(v ? { job: v } : {})} /></Field>
      </Card>
      {!jobId ? <Card><Empty icon={<Target className="h-6 w-6" />} title="Choose a job to rank the candidates" /></Card>
        : ranking.isLoading ? <Loading label="Grading every candidate (BGE-M3 + reranker)…" rows={6} />
        : ranking.error ? <ErrorState error={ranking.error} /> : (
          <Card>
            <CardHeader title={`${rows.length} candidates ranked`} subtitle="Fast semantic estimate: must-haves met / partial / missing from CV evidence" icon={<Zap className="h-4 w-4" />} />
            <div className="overflow-x-auto p-3">
              <table className="table-x">
                <thead><tr><th>#</th><th>Candidate</th><th>Fast grade</th><th>Score</th><th>Must-haves</th><th>Experience</th><th>Gaps</th><th>Analysis</th><th></th></tr></thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={r.candidate_id}>
                      <td className="muted font-mono">{i + 1}</td>
                      <td className="font-medium">{r.name || "Unnamed"}</td>
                      <td><GradeBadge grade={r.grade} label={r.grade_label} /></td>
                      <td className="w-32"><div className="text-xs font-semibold">{r.score}</div><Progress value={r.score / 100} className="mt-1" /></td>
                      <td className="text-xs"><span className="text-emerald-600 dark:text-emerald-300">{r.met} met</span> · <span className="text-amber-600 dark:text-amber-300">{r.partial} partial</span> · <span className="text-rose-600 dark:text-rose-300">{r.missing} missing</span><div className="muted">coverage {pct(r.must_coverage)}</div></td>
                      <td className="text-xs">{r.total_years ?? "?"} y</td>
                      <td className="max-w-[220px] text-xs">{(r.gaps || []).join("; ") || "—"}</td>
                      <td>{r.analysis_id ? <button className="flex items-center gap-1.5" onClick={() => setReport(r.analysis_id)}><GradeBadge grade={r.analysis_grade} /><Badge tone={ANALYSIS_STATUS[r.analysis_status]?.tone}>{ANALYSIS_STATUS[r.analysis_status]?.label}</Badge></button> : <span className="muted text-xs">—</span>}</td>
                      <td className="whitespace-nowrap text-right">
                        <button className="btn-primary btn-sm" disabled={analyse.isPending} onClick={() => analyse.mutate({ candidate: r.candidate_id, method: "llm" })} title="Qwen2.5-7B + RAG, about a minute on the Kaggle GPU">
                          {analyse.isPending && analyse.variables?.candidate === r.candidate_id ? <Spinner /> : <Sparkles className="h-3.5 w-3.5" />} AI analysis
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!rows.length && <Empty title="No candidate to rank" />}
            </div>
          </Card>
        )}
      <Drawer open={!!report} onClose={() => setReport(null)} width="max-w-5xl" title={analysis.data ? `${analysis.data.candidate_name} — ${analysis.data.job_title}` : "Analysis"}
        actions={analysis.data?.status === "pending_review" && <Link to={`/validation?id=${report}`} className="btn-primary btn-sm">Decide</Link>}>
        {analysis.isLoading || !analysis.data ? <Loading /> : <ReportView analysis={analysis.data} />}
      </Drawer>
    </div>
  );
}
