import { useMutation } from "@tanstack/react-query";
import { BrainCircuit, Play } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { CandidateSelect, JobSelect } from "../components/common";
import { DocLink } from "../components/domain/DocViewer";
import { Badge, Card, CardHeader, CodeBlock, Collapsible, Field, GradeBadge, KeyVals, PageHeader, Spinner } from "../components/ui";
import { api, errorText } from "../lib/api";
import { LEVELS, REQ_STATUS } from "../lib/format";

export default function MatchingInternals() {
  const [candidate, setCandidate] = useState("");
  const [job, setJob] = useState("");
  const run = useMutation({ mutationFn: () => api.post("/observe/matching", { candidate_id: candidate, job_id: job }), onError: (e) => toast.error(errorText(e)) });
  const r = run.data;
  return (
    <div>
      <PageHeader eyebrow="Diagram part 5 · contextualization" title="Matching internals" icon={<BrainCircuit className="h-7 w-7" />} subtitle="Everything that feeds an analysis, without calling the LLM: the CV evidence found for every requirement (BGE-M3 cosine, BGE reranker), the HR rules the RAG retrieves for the job, and the exact prompt Qwen receives." />
      <Card className="mb-6 p-5">
        <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
          <Field label="Candidate"><CandidateSelect value={candidate} onChange={setCandidate} /></Field>
          <Field label="Job"><JobSelect value={job} onChange={setJob} /></Field>
          <button className="btn-primary" disabled={!candidate || !job || run.isPending} onClick={() => run.mutate()}>{run.isPending ? <Spinner /> : <Play className="h-4 w-4" />} Show the internals</button>
        </div>
      </Card>
      {r && (
        <div className="space-y-5">
          <Card className="p-5"><KeyVals cols={4} items={[["Pair", `${r.candidate} → ${r.job}`], ["CV snippets", r.snippet_count], ["Thresholds", `demonstrated ≥ ${r.thresholds.demonstrated} · equivalent ≥ ${r.thresholds.equivalent}`], ["Fast grade", <GradeBadge key="g" grade={r.semantic_grade} />]]} /></Card>
          <Card>
            <CardHeader title="Evidence per requirement" subtitle={r.reranked ? "Scores: BGE reranker (cross-encoder), cosine shown for reference" : "Scores: cosine similarity (no reranker)"} />
            <div className="space-y-3 p-5">
              {r.requirements.map((q: any) => (
                <Collapsible key={q.id} title={<span className="flex flex-wrap items-center gap-2 text-sm"><Badge tone={q.kind === "must" ? "indigo" : "slate"}>{q.id}</Badge>{q.text}<Badge tone={REQ_STATUS[q.status]?.tone}>{REQ_STATUS[q.status]?.label}</Badge><span className="muted text-xs">{LEVELS[q.level]}</span></span>}>
                  <div className="space-y-1.5">{q.snippets.map((s: any, i: number) => (
                    <div key={i} className="flex gap-3 rounded-lg surface-2 p-2 text-xs"><span className="w-28 shrink-0 font-mono">{s.rerank ?? "—"} <span className="muted">/ cos {s.cosine}</span></span><span className="muted w-40 shrink-0 truncate">{s.where}</span><span>{s.text}</span></div>
                  ))}{q.note && <div className="muted text-xs">{q.note}</div>}</div>
                </Collapsible>
              ))}
            </div>
          </Card>
          <Card>
            <CardHeader title="HR rules retrieved by the RAG" subtitle={`In the prompt: ${r.rules_in_prompt.join(", ") || "none"}`} />
            <div className="space-y-3 p-5">
              {r.knowledge.map((k: any, i: number) => (
                <div key={i} className="rounded-xl border hairline p-3">
                  <div className="mb-2 text-xs"><span className="muted">query:</span> <b>“{k.query}”</b>{k.job_profile && <Badge tone="cyan" className="ml-2">{k.job_profile}</Badge>}{k.error && <span className="ml-2 text-rose-500">{k.error}</span>}</div>
                  <div className="space-y-1.5">{k.hits.map((h: any, j: number) => <div key={j} className="text-xs"><DocLink docId={h.doc_id} section={h.section}>{h.doc_id}</DocLink> <span className="muted">{h.section} · {h.score}</span><div className="muted line-clamp-2">{h.text}</div></div>)}</div>
                </div>
              ))}
            </div>
          </Card>
          <Collapsible title={`The exact prompt for Qwen (${r.prompt_chars} characters)`}><div className="space-y-3"><CodeBlock title="system" code={r.system_prompt} maxHeight="max-h-60" /><CodeBlock title="user" code={r.prompt} /></div></Collapsible>
        </div>
      )}
    </div>
  );
}
