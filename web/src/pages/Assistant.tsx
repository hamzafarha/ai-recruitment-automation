import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowRight, Bot, Brain, ChevronRight, MessageSquare, SendHorizontal, Sparkles, User, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { DocLink } from "../components/domain/DocViewer";
import { Badge, Card, CardHeader, CodeBlock, Collapsible, cx, Empty, JsonView, Markdown, PageHeader, Spinner } from "../components/ui";
import { api, errorText } from "../lib/api";
import { ago, num } from "../lib/format";

const EXAMPLES = [
  "Who are the best candidates for the ML Engineer job?",
  "Which candidates have production experience with Kubernetes?",
  "How are internships counted in years of experience?",
  "Compare Priya Raman with the GenAI Engineer job.",
  "What does the fair evaluation policy forbid?",
];

function Trace({ run }: { run: any }) {
  return (
    <Collapsible className="mt-3" title={<span className="flex items-center gap-2 text-xs"><Brain className="h-3.5 w-3.5 text-indigo-500" /> How the agent answered · {run.steps.length} tool call(s) · {run.llm_calls} Qwen call(s) · {num(run.seconds, 1)} s · {run.graph}</span>}>
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-1.5">
          {(run.timeline || []).map((e: any, i: number) => (
            <span key={i} className="flex items-center gap-1.5">
              {i > 0 && <ArrowRight className="muted h-3 w-3" />}
              <Badge tone={e.node === "act" ? "cyan" : e.node === "respond" ? "emerald" : "indigo"} title={e.why || e.result}>
                {e.node}{e.tool ? `: ${e.tool}` : ""}{e.seconds !== undefined ? ` · ${num(e.seconds, 1)}s` : ""}
              </Badge>
            </span>
          ))}
        </div>
        {run.steps.map((s: any, i: number) => (
          <div key={i} className="rounded-xl border hairline p-3">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Badge tone="cyan"><Wrench className="h-3 w-3" />{s.tool}</Badge>
              <Badge tone={s.ok ? "emerald" : "rose"}>{s.ok ? "ok" : "error"}</Badge>
              <span className="muted">{num(s.seconds, 2)} s</span>
            </div>
            {s.thought && <div className="mt-1.5 text-[13px] italic">“{s.thought}”</div>}
            <div className="mt-2 font-mono text-[11px]">{s.tool}({JSON.stringify(s.arguments)})</div>
            {s.error && <div className="mt-1 text-xs text-rose-600">{s.error}</div>}
            {s.tool === "search_hr_knowledge" && Array.isArray(s.data) && (
              <div className="mt-2 space-y-1.5">{s.data.map((d: any, j: number) => <div key={j} className="rounded-lg surface-2 p-2 text-xs"><DocLink docId={d.doc_id} section={d.section}>{d.doc_id}</DocLink> <span className="muted">· {d.section} · relevance {d.relevance}</span><div className="muted mt-1 line-clamp-2">{d.text}</div></div>)}</div>
            )}
            {s.data !== undefined && s.data !== null && <div className="mt-2"><Collapsible title={<span className="text-xs">What the tool returned (given to Qwen)</span>}><JsonView data={s.data} maxHeight="max-h-72" /></Collapsible></div>}
          </div>
        ))}
        {!!run.sources?.length && (
          <div className="flex flex-wrap items-center gap-2 text-xs"><span className="muted">Sources:</span>
            {run.sources.map((s: any, i: number) => s.type === "knowledge" ? <DocLink key={i} docId={s.id}>{s.title}</DocLink> : <Badge key={i} tone={s.type === "job" ? "violet" : "indigo"}>{s.type}: {s.title}</Badge>)}
          </div>
        )}
        {run.warnings?.map((w: string, i: number) => <div key={i} className="text-xs text-amber-600">{w}</div>)}
      </div>
    </Collapsible>
  );
}

export default function Assistant() {
  const client = useQueryClient();
  const [question, setQuestion] = useState("");
  const end = useRef<HTMLDivElement>(null);
  const runs = useQuery({ queryKey: ["agent-runs"], queryFn: () => api.get("/agent/runs?limit=30").then((r) => (r.data as any[]).slice().reverse()) });
  const tools = useQuery({ queryKey: ["agent-tools"], queryFn: () => api.get("/agent/tools") });
  const ask = useMutation({
    mutationFn: (q: string) => api.post("/agent/ask", { question: q }),
    onSuccess: () => { setQuestion(""); client.invalidateQueries({ queryKey: ["agent-runs"] }); },
    onError: (e) => toast.error(errorText(e)),
  });
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [runs.data?.length, ask.isPending]);
  const send = (q: string) => q.trim().length >= 3 && !ask.isPending && ask.mutate(q.trim());
  return (
    <div>
      <PageHeader eyebrow="DSO4 · diagram part 5" title="AI assistant" icon={<Bot className="h-7 w-7" />}
        subtitle="A LangGraph agent (decide → act → respond) with 8 tools over PostgreSQL, Qdrant, the reranker and the matcher. It chooses its tools, runs them, then Qwen answers from their results only. Every answer shows the tools called, their arguments, what they returned and the documents used." />
      <div className="grid gap-5 xl:grid-cols-[1fr_340px]">
        <Card className="flex min-h-[70vh] flex-col">
          <div className="flex-1 space-y-6 overflow-y-auto p-5">
            {!runs.data?.length && !ask.isPending && (
              <Empty icon={<MessageSquare className="h-6 w-6" />} title="Ask about candidates, jobs or HR rules">
                <div className="mt-3 flex flex-wrap justify-center gap-2">{EXAMPLES.map((e) => <button key={e} onClick={() => send(e)} className="chip hover:border-indigo-400">{e}</button>)}</div>
              </Empty>
            )}
            {(runs.data || []).map((run) => (
              <motion.div key={run.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                <div className="flex justify-end gap-3"><div className="max-w-[80%] rounded-2xl rounded-tr-md bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-2.5 text-sm text-white shadow-lg shadow-indigo-500/20">{run.question}</div><div className="grid h-8 w-8 shrink-0 place-items-center rounded-full surface-2"><User className="h-4 w-4" /></div></div>
                <div className="flex gap-3">
                  <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-indigo-500 to-cyan-400 text-white"><Sparkles className="h-4 w-4" /></div>
                  <div className="min-w-0 max-w-[88%] flex-1 rounded-2xl rounded-tl-md border hairline px-4 py-3">
                    <Markdown className="text-sm">{run.answer || "—"}</Markdown>
                    <div className="muted mt-1 text-[10px]">{ago(run.created_at)} · {run.model?.split("/").pop()}</div>
                    <Trace run={run} />
                  </div>
                </div>
              </motion.div>
            ))}
            {ask.isPending && (
              <div className="space-y-3">
                <div className="flex justify-end"><div className="max-w-[80%] rounded-2xl rounded-tr-md bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-2.5 text-sm text-white">{ask.variables}</div></div>
                <div className="muted flex items-center gap-2 text-sm"><Spinner /> The agent decides which tool to use, runs it, then writes the answer (about a minute on the Kaggle GPU)…</div>
              </div>
            )}
            <div ref={end} />
          </div>
          <div className="border-t hairline p-4">
            <form onSubmit={(e) => { e.preventDefault(); send(question); }} className="flex gap-2">
              <input className="input" placeholder="Ask the assistant… (e.g. Who has RAG experience?)" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={1000} />
              <button className="btn-primary" disabled={question.trim().length < 3 || ask.isPending}>{ask.isPending ? <Spinner /> : <SendHorizontal className="h-4 w-4" />}</button>
            </form>
          </div>
        </Card>
        <div className="space-y-5">
          <Card>
            <CardHeader title="The agent's tools" subtitle="What Qwen can call (the code validates every argument)" icon={<Wrench className="h-4 w-4" />} />
            <div className="space-y-2 p-4">
              {(tools.data?.data || []).map((t: any) => (
                <div key={t.name} className="rounded-xl border hairline p-3">
                  <div className="font-mono text-xs font-semibold text-indigo-600 dark:text-indigo-300">{t.signature}</div>
                  <div className="muted mt-1 text-xs leading-relaxed">{t.description}</div>
                </div>
              ))}
            </div>
          </Card>
          {tools.data?.system_prompts && (
            <Card className="p-4">
              <Collapsible title={<><ChevronRight className="h-4 w-4" /> System prompts (decide, respond)</>}>
                <div className="space-y-3"><CodeBlock title="decide" code={tools.data.system_prompts.decide} maxHeight="max-h-64" /><CodeBlock title="respond" code={tools.data.system_prompts.respond} maxHeight="max-h-64" /></div>
              </Collapsible>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

