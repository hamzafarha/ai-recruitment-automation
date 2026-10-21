// The application pipeline: the 12 steps as a live stepper, and what every step produced, in full.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  BellRing, Bot, Check, Database, FileText, Gavel, Inbox, Lock, Mail, ScanText, Send, Sparkles, UserCheck, X,
} from "lucide-react";
import { ReactNode, useMemo, useState } from "react";
import { toast } from "sonner";
import { api, errorText } from "../../lib/api";
import { DECISIONS, secs, when } from "../../lib/format";
import { Badge, Callout, CodeBlock, Collapsible, cx, Empty, KeyVals, Spinner, Tabs } from "../ui";
import { AgentGraph, AgentJobTrace, AgentTimeline, ToolCalls } from "./Agent";
import { DecisionForm } from "./DecisionForm";
import { DocLink } from "./DocViewer";
import { EmailCard, OutboxHint } from "./Email";
import { ProfileCompare, ProfileView } from "./Profile";
import { ReportView } from "./Report";

export const STEP_ICONS: Record<string, ReactNode> = {
  intake: <Inbox className="h-4 w-4" />, document: <ScanText className="h-4 w-4" />, extract: <Sparkles className="h-4 w-4" />,
  store: <Database className="h-4 w-4" />, acknowledge: <Mail className="h-4 w-4" />, screen: <Bot className="h-4 w-4" />,
  report: <FileText className="h-4 w-4" />, notify_recruiters: <BellRing className="h-4 w-4" />, validation: <UserCheck className="h-4 w-4" />,
  decision: <Gavel className="h-4 w-4" />, notify_candidate: <Send className="h-4 w-4" />, close: <Lock className="h-4 w-4" />,
};
const STATUS_STYLE: Record<string, string> = {
  done: "border-emerald-400/70 bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
  running: "border-sky-400 bg-sky-500/15 text-sky-600 dark:text-sky-300 animate-pulse-ring",
  waiting: "border-amber-400 bg-amber-500/15 text-amber-600 dark:text-amber-300 animate-pulse-ring",
  failed: "border-rose-400 bg-rose-500/15 text-rose-600 dark:text-rose-300",
  pending: "hairline bg-slate-500/5 text-slate-400",
};

export function PipelineStepper({ steps, selected, onSelect }: { steps: any[]; selected: string; onSelect: (s: string) => void }) {
  return (
    <div className="overflow-x-auto pb-2">
      <div className="flex min-w-max items-start px-1 pt-3">
        {steps.map((s, i) => {
          const status = s.status || "pending";
          const next = steps[i + 1];
          return (
            <div key={s.step} className="flex items-start">
              <button onClick={() => onSelect(s.step)} className="group flex w-[92px] flex-col items-center text-center">
                <motion.div initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: i * 0.03 }}
                  className={cx("relative grid h-11 w-11 place-items-center rounded-2xl border-2 transition group-hover:scale-105", STATUS_STYLE[status], selected === s.step && "ring-4 ring-indigo-500/25")}>
                  {STEP_ICONS[s.step]}
                  {status === "done" && <span className="absolute -bottom-1 -right-1 grid h-4 w-4 place-items-center rounded-full bg-emerald-500 text-white"><Check className="h-2.5 w-2.5" /></span>}
                  {status === "failed" && <span className="absolute -bottom-1 -right-1 grid h-4 w-4 place-items-center rounded-full bg-rose-500 text-white"><X className="h-2.5 w-2.5" /></span>}
                </motion.div>
                <div className={cx("mt-2 text-[11px] font-semibold leading-tight", selected === s.step ? "text-indigo-600 dark:text-indigo-300" : "")}>{s.label}</div>
                <div className="muted mt-0.5 text-[10px]">part {s.part}{s.seconds !== undefined && s.seconds !== null && status === "done" ? ` · ${secs(s.seconds)}` : status !== "pending" && status !== "done" ? ` · ${status}` : ""}</div>
              </button>
              {next && (
                <div className="mt-[21px] h-[3px] w-6 overflow-hidden rounded-full bg-slate-500/15">
                  {status === "done" && (next.status === "running" || next.status === "waiting")
                    ? <div className="flow-line h-full w-full" />
                    : <div className={cx("h-full", status === "done" ? "w-full bg-emerald-400/70" : "w-0")} />}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ step panels

function DocumentStep({ result, outputs }: { result: any; outputs: any }) {
  return (
    <div className="space-y-4">
      <KeyVals cols={4} items={[["File type", result.file_type], ["Reader", result.engine], ["Input", result.input_type], ["Pages", `${result.pages} (${result.ocr_pages} by OCR)`], ["Characters", result.characters], ["Reading time", secs(result.seconds)]]} />
      {!!(result.page_details || []).length && (
        <div className="overflow-x-auto rounded-xl border hairline">
          <table className="table-x"><thead><tr><th>Page</th><th>Method</th><th>Characters</th><th>OCR confidence</th></tr></thead>
            <tbody>{result.page_details.map((p: any) => <tr key={p.page}><td>{p.page}</td><td><Badge tone={p.method === "ocr" ? "amber" : "sky"}>{p.method}</Badge></td><td>{p.characters}</td><td>{p.ocr_confidence ?? "—"}</td></tr>)}</tbody>
          </table>
        </div>
      )}
      <CodeBlock title={`CV text read · ${outputs?.document?.characters ?? result.characters} characters (the file itself was deleted, KB-HR-004)`} code={outputs?.document?.cv_text || result.preview || ""} maxHeight="max-h-[420px]" />
    </div>
  );
}

function ExtractStep({ result, outputs }: { result: any; outputs: any }) {
  const client = useQueryClient();
  const candidateId = outputs?.extract?.candidate_id;
  const compare = useMutation({
    mutationFn: () => api.post(`/candidates/${candidateId}/compare-models`),
    onSuccess: () => { toast.success("Base model compared"); client.invalidateQueries({ queryKey: ["outputs"] }); },
    onError: (e) => toast.error(errorText(e)),
  });
  const comparison = outputs?.extract?.model_comparison;
  const [view, setView] = useState<"profile" | "compare">("profile");
  return (
    <div className="space-y-4">
      <KeyVals cols={4} items={[
        ["Model", result.model], ["QLoRA adapter", result.adapter ? <Badge tone="indigo">{String(result.adapter).split("/").pop()}</Badge> : <Badge tone="slate">none (mock / base)</Badge>],
        ["Tokens", result.prompt_tokens ? `${result.prompt_tokens} → ${result.output_tokens}` : "—"], ["Generation", secs(result.seconds)],
      ]} />
      {result.input_truncated && <Callout tone="amber">The CV was longer than the model's input limit: its end was cut.</Callout>}
      {result.contact_fixes && (
        <Callout tone="amber">
          Checked against the CV text by code:{" "}
          {Object.entries(result.contact_fixes as Record<string, { model: string | null; cv: string | null }>).map(([field, f], i) => (
            <span key={field}>{i > 0 && "; "}{field} — the model wrote <b>{f.model || "nothing"}</b>, {f.cv ? <>the CV says <b>{f.cv}</b></> : "it is nowhere in the CV (removed)"}</span>
          ))}.
        </Callout>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Tabs className="mb-0" value={view} onChange={setView} tabs={[{ id: "profile", label: "Extracted CV (fine-tuned Qwen)" }, { id: "compare", label: "Base Qwen2.5 vs + QLoRA" }]} />
        <button className="btn-ghost btn-sm" disabled={!candidateId || compare.isPending} onClick={() => { compare.mutate(); setView("compare"); }}
          title={candidateId ? "Runs the base model (no adapter) on this CV: about a minute on the Kaggle GPU" : "Available once the candidate is stored"}>
          {compare.isPending ? <Spinner /> : <Sparkles className="h-3.5 w-3.5" />} {comparison ? "Compare again" : "Compare with the base model"}
        </button>
      </div>
      {view === "profile" ? <ProfileView profile={outputs?.extract?.profile} /> : comparison ? (
        <div><div className="muted mb-3 text-xs">Compared {when(comparison.at)} · same CV text, same prompt and code: only the adapter differs.</div><ProfileCompare result={comparison} /></div>
      ) : (
        <Empty icon={<Sparkles className="h-6 w-6" />} title="Not compared yet">Run the base Qwen2.5-7B (without our QLoRA adapter) on this same CV to see what the fine-tuning changes, field by field.</Empty>
      )}
    </div>
  );
}

function StoreStep({ result, outputs }: { result: any; outputs: any }) {
  const q = outputs?.store?.qdrant;
  return (
    <div className="space-y-4">
      <KeyVals cols={3} items={[["Candidate", result.full_name], ["PostgreSQL", `table candidates · id ${String(result.candidate_id).slice(0, 8)}`], ["Search index", result.indexed ? <Badge tone="emerald" dot>indexed</Badge> : <Badge tone="rose">not indexed: {result.index_error}</Badge>]]} />
      <div>
        <div className="mb-2 text-sm font-semibold">Qdrant points of this candidate <span className="muted font-normal">— collection “{q?.collection || "candidates"}”, one BGE-M3 vector per snippet ({q?.count ?? 0})</span></div>
        {q?.error && <Callout tone="amber">{q.error}</Callout>}
        <div className="grid gap-2 md:grid-cols-2">
          {(q?.points || []).map((p: any) => (
            <div key={p.id} className="rounded-xl border hairline p-3 text-xs">
              <div className="mb-1 flex items-center gap-2"><Badge tone="cyan">{p.section}</Badge><span className="muted truncate">{p.label}</span></div>
              <div className="line-clamp-3 leading-relaxed">{p.text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ScreenStep({ result, outputs }: { result: any; outputs: any }) {
  const [node, setNode] = useState<string | null>(null);
  const jobs: any[] = outputs?.screen?.jobs || [];
  const [job, setJob] = useState(0);
  const docs = useMemo(() => {
    const seen = new Map<string, any>();
    jobs.forEach((j) => (j.knowledge_sources || []).forEach((k: any) => { const key = `${k.doc_id}|${k.section}`; if (!seen.has(key)) seen.set(key, k); }));
    return [...seen.values()];
  }, [jobs]);
  return (
    <div className="space-y-5">
      <KeyVals cols={4} items={[["Task", result.task], ["Qwen calls", result.llm_calls], ["Tool calls", result.tool_calls], ["Time", secs(result.seconds)], ["CV passages", `${result.passages} (from the ${result.passages_from === "cv_text" ? "CV text" : "profile"})`], ["Graph", result.graph], ["Model", result.model || "—"], ["Jobs analysed", (result.jobs_analysed || []).length]]} />
      {!!(result.jobs_considered || []).length && (
        <div>
          <div className="mb-2 text-sm font-semibold">1 · Understand the task — open jobs graded by the Semantic Matching tool</div>
          <div className="flex flex-wrap gap-2">{result.jobs_considered.map((c: any) => <Badge key={c.job_id} tone={c.grade >= 2 ? "emerald" : "slate"}>{c.title} · grade {c.grade} · {c.score}</Badge>)}</div>
        </div>
      )}
      <div>
        <div className="mb-2 flex items-center justify-between text-sm font-semibold">
          <span>The path through the LangGraph graph <span className="muted font-normal">— click a node to filter the timeline</span></span>
          {node && <button className="btn-ghost btn-sm" onClick={() => setNode(null)}>Show all</button>}
        </div>
        <AgentGraph timeline={result.timeline || []} selected={node} onSelect={setNode} />
      </div>
      <Collapsible defaultOpen title={<>Timeline — every node run, in order ({(result.timeline || []).length})</>}>
        <AgentTimeline timeline={result.timeline || []} filter={node} />
      </Collapsible>
      <Collapsible title={<>All tools called ({(result.tools || []).length}), with their inputs and results</>}>
        <ToolCalls calls={result.tools || []} />
      </Collapsible>
      <div>
        <div className="mb-2 text-sm font-semibold">Documents the agent retrieved from the HR knowledge base (RAG) <span className="muted font-normal">— click to open the exact document on the section used</span></div>
        {docs.length ? (
          <div className="flex flex-wrap gap-2">{docs.map((k, i) => <span key={i} className="chip"><DocLink docId={k.doc_id} section={k.section} chunkId={k.chunk_id}>{k.doc_id}</DocLink><span className="muted">{k.section}</span></span>)}</div>
        ) : <div className="muted text-sm">No HR document retrieved (index the knowledge base: Dashboard → Set up).</div>}
      </div>
      {(result.warnings || []).map((w: string, i: number) => <Callout key={i} tone="amber">{w}</Callout>)}
      {!!jobs.length && (
        <div>
          <div className="mb-2 text-sm font-semibold">Per job: planning, evidence, checks, prompt</div>
          {jobs.length > 1 && <Tabs value={String(job)} onChange={(v) => setJob(Number(v))} tabs={jobs.map((j, i) => ({ id: String(i), label: `${j.job_title} · grade ${j.grade}` }))} />}
          {jobs[job] && <AgentJobTrace job={jobs[job]} systemPrompts={outputs?.screen?.system_prompts} />}
        </div>
      )}
    </div>
  );
}

function EmailsStep({ emails, result }: { emails: any[]; result: any }) {
  if (!emails?.length) {
    return result?.text ? (
      <div className="space-y-2"><div className="text-sm"><b>To:</b> {result.to || "(no e-mail in the CV)"} · <b>Subject:</b> {result.subject}</div><pre className="whitespace-pre-wrap rounded-xl border hairline p-3 font-sans text-[13px]">{result.text}</pre></div>
    ) : <div className="muted text-sm">No e-mail written for this step.</div>;
  }
  return <div className="space-y-3"><OutboxHint status={emails[0]?.status} />{emails.map((e) => <EmailCard key={e.id} email={e} />)}</div>;
}

function ValidationStep({ app, step, outputs }: { app: any; step: any; outputs: any }) {
  const reviews: any[] = outputs?.validation?.reviews || [];
  const pending = (outputs?.report?.analyses || []).filter((a: any) => a.status === "pending_review" || a.status === "on_hold");
  if (step.status === "pending" && !reviews.length) return <Empty icon={STEP_ICONS.validation} title="Not reached yet">{step.what}</Empty>;
  return (
    <div className="space-y-4">
      {step.status === "waiting" && (
        <>
          <Callout tone="amber" title="The workflow waits for a recruiter's decision (KB-RR-001)">Decide here or in Human validation: the decision resumes this application's workflow (final decision → e-mail to the candidate → closed){app.orchestrator === "n8n" ? ", through the n8n Wait node" : ""}.</Callout>
          {pending.map((a: any) => (
            <div key={a.id} className="rounded-2xl border hairline p-4">
              <div className="mb-3 text-sm font-semibold">{a.job_title} · grade {a.grade} · {a.score}/100</div>
              <DecisionForm analysis={a} />
            </div>
          ))}
        </>
      )}
      {reviews.map((r, i) => (
        <div key={i} className="flex flex-wrap items-center gap-3 rounded-xl border hairline p-3 text-sm">
          <Badge tone={DECISIONS[r.decision]?.tone}>{DECISIONS[r.decision]?.label || r.decision}</Badge>
          <span>{r.job_title}</span><span className="muted">by {r.reviewer} · {when(r.created_at)}</span>
          {r.interview_at && <Badge tone="violet">interview {r.interview_at}</Badge>}
          {r.adjusted_grade !== null && r.adjusted_grade !== undefined && <Badge tone="slate">grade → {r.adjusted_grade}</Badge>}
          {r.notes && <span className="muted text-xs italic">“{r.notes}” (internal)</span>}
        </div>
      ))}
      {step.note && <Callout tone="sky">{step.note}</Callout>}
    </div>
  );
}

export function StepPanel({ app, step, outputs }: { app: any; step: any; outputs: any }) {
  const result = step.result || {};
  const name = step.step;
  let body: ReactNode;
  if (step.status === "pending" && name !== "validation") body = <Empty icon={STEP_ICONS[name]} title="Not run yet">{step.what}</Empty>;
  else if (name === "document" && step.result) body = <DocumentStep result={result} outputs={outputs} />;
  else if (name === "extract" && step.result) body = <ExtractStep result={result} outputs={outputs} />;
  else if (name === "store" && step.result) body = <StoreStep result={result} outputs={outputs} />;
  else if (name === "acknowledge") body = <EmailsStep emails={outputs?.acknowledge?.emails} result={result} />;
  else if (name === "screen" && step.result) body = <ScreenStep result={result} outputs={outputs} />;
  else if (name === "report") {
    const analyses = outputs?.report?.analyses || [];
    body = analyses.length ? <div className="space-y-6">{analyses.map((a: any) => <div key={a.id}><div className="mb-2 font-display font-semibold">{a.job_title}</div><ReportView analysis={a} /></div>)}</div> : <div className="muted text-sm">No report.</div>;
  } else if (name === "notify_recruiters" && step.result) body = (
    <div className="space-y-3">
      <KeyVals items={[["Notification", result.title], ["Priority", result.priority ? <Badge tone="amber">priority (grade 2-3)</Badge> : "normal"], ["Waiting for", result.waiting_for], ["Resumed by", result.resumed_by]]} />
      {(outputs?.notify_recruiters?.notifications || []).map((n: any) => <div key={n.id} className="rounded-xl border hairline p-3 text-sm"><div className="font-medium">{n.title}</div><div className="muted mt-1 whitespace-pre-wrap text-xs">{n.message}</div></div>)}
    </div>
  );
  else if (name === "validation") body = <ValidationStep app={app} step={step} outputs={outputs} />;
  else if (name === "decision" && step.result) body = result.decided ? (
    <KeyVals cols={4} items={[["Decision", <Badge key="d" tone={DECISIONS[result.decision]?.tone}>{DECISIONS[result.decision]?.label || result.decision}</Badge>], ["Job", result.job_title], ["Reviewer", result.reviewer], ["Interview", result.interview_at || "—"]]} />
  ) : <div className="muted text-sm">No final decision yet (on hold, or another analysis waits).</div>;
  else if (name === "notify_candidate") body = <EmailsStep emails={outputs?.notify_candidate?.emails} result={result} />;
  else if (name === "close" && step.result) body = <KeyVals items={[["Final decision", DECISIONS[result.decision]?.label || result.decision], ["From CV to close", secs(result.total_seconds)]]} />;
  else body = step.result ? <CodeBlock title="result" code={JSON.stringify(step.result, null, 2)} /> : <div className="muted text-sm">—</div>;
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 font-display text-lg font-semibold"><span className="text-indigo-500">{STEP_ICONS[name]}</span>{step.label}<Badge tone="slate">part {step.part}</Badge></div>
          <div className="muted mt-1 text-sm">{step.what}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge tone={{ done: "emerald", running: "sky", waiting: "amber", failed: "rose" }[step.status as string] || "slate"} dot>{step.status}</Badge>
          {step.seconds !== undefined && step.seconds !== null && <span className="muted">{secs(step.seconds)}</span>}
          {step.finished_at && <span className="muted">finished {when(step.finished_at)}</span>}
        </div>
      </div>
      {step.error && <Callout tone="rose" title="This step failed" className="mb-4">{step.error}</Callout>}
      {step.status === "running" && <div className="muted mb-4 flex items-center gap-2 text-sm"><Spinner /> Running… (the page refreshes by itself)</div>}
      {body}
    </div>
  );
}

