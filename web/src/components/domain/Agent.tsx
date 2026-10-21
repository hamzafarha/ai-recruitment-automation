// The screening agent (LangGraph) made visible: its graph with the path it took, every tool call with what
// it returned, the documents it retrieved, its planning rounds, the prompt, Qwen's answer and the checks.
import { Background, Controls, Edge, Handle, MarkerType, Node, NodeProps, Position, ReactFlow } from "@xyflow/react";
import { Brain, CheckCircle2, ChevronRight, Compass, FileSearch, Flag, Gauge, ListChecks, Play, Repeat, Search, ShieldCheck, Sparkles, Wrench } from "lucide-react";
import { ReactNode, useMemo, useState } from "react";
import { num } from "../../lib/format";
import { Badge, CodeBlock, Collapsible, cx, KeyVals } from "../ui";
import { DocLink } from "./DocViewer";

// ------------------------------------------------------------------ the LangGraph graph

const NODE_INFO: Record<string, { label: string; step: string; icon: ReactNode; x: number; y: number }> = {
  start: { label: "START", step: "", icon: <Play className="h-3.5 w-3.5" />, x: 0, y: 120 },
  understand: { label: "understand", step: "1 · Understand the task", icon: <Compass className="h-3.5 w-3.5" />, x: 120, y: 120 },
  plan: { label: "plan", step: "2 · Decide which tools", icon: <Brain className="h-3.5 w-3.5" />, x: 320, y: 120 },
  act: { label: "act", step: "3-4 · Call tools, retrieve", icon: <Wrench className="h-3.5 w-3.5" />, x: 520, y: 120 },
  evaluate: { label: "evaluate", step: "5-6 · Rerank, iterate?", icon: <Gauge className="h-3.5 w-3.5" />, x: 720, y: 120 },
  analyse: { label: "analyse", step: "7 · Qwen judgement", icon: <Sparkles className="h-3.5 w-3.5" />, x: 920, y: 120 },
  verify: { label: "verify", step: "7 · Code checks claims", icon: <ShieldCheck className="h-3.5 w-3.5" />, x: 1120, y: 120 },
  next_job: { label: "next_job", step: "Another job?", icon: <Repeat className="h-3.5 w-3.5" />, x: 1120, y: 290 },
  end: { label: "END", step: "", icon: <Flag className="h-3.5 w-3.5" />, x: 1320, y: 290 },
};
const EDGES: [string, string, string, string, string?][] = [
  ["start", "understand", "right-s", "left-t"],
  ["understand", "plan", "right-s", "left-t"],
  ["plan", "act", "right-s", "left-t"],
  ["act", "evaluate", "right-s", "left-t"],
  ["evaluate", "plan", "top-s", "top-t", "weak must-haves"],
  ["evaluate", "analyse", "right-s", "left-t"],
  ["analyse", "verify", "right-s", "left-t"],
  ["analyse", "next_job", "bottom-s", "left-t", "model failed"],
  ["verify", "next_job", "bottom-s", "top-t"],
  ["next_job", "plan", "bottom-s", "bottom-t", "next job"],
  ["next_job", "end", "right-s", "left-t"],
];

const SIDES: [string, Position][] = [["left", Position.Left], ["right", Position.Right], ["top", Position.Top], ["bottom", Position.Bottom]];

function GraphNode({ data }: NodeProps) {
  const d = data as any;
  const terminal = d.kind === "start" || d.kind === "end";
  return (
    <div onClick={d.onClick}
      className={cx("relative cursor-pointer rounded-xl border px-3 py-2 text-xs transition",
        terminal ? "rounded-full px-3 py-1.5" : "min-w-[150px]",
        d.visits ? "border-indigo-400/60 bg-white shadow-[0_0_0_4px_rgb(99_102_241/0.12),0_10px_30px_-10px_rgb(99_102_241/0.6)] dark:bg-[#141b33]" : "hairline bg-white/60 opacity-50 dark:bg-white/[0.02]",
        d.selected && "ring-2 ring-amber-400")}>
      {SIDES.map(([side, pos]) => (
        <span key={side}>
          <Handle type="source" id={`${side}-s`} position={pos} />
          <Handle type="target" id={`${side}-t`} position={pos} />
        </span>
      ))}
      <div className="flex items-center gap-1.5 font-mono font-semibold">
        <span className={d.visits ? "text-indigo-500 dark:text-indigo-300" : "muted"}>{d.icon}</span>{d.label}
      </div>
      {!terminal && <div className="muted mt-0.5 text-[10px]">{d.step}</div>}
      {d.visits > 0 && !terminal && (
        <div className="absolute -right-2 -top-2 flex items-center gap-1 rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 px-1.5 py-0.5 text-[10px] font-bold text-white shadow">
          ×{d.visits}{d.seconds ? <span className="font-normal opacity-80">· {num(d.seconds, 1)}s</span> : null}
        </div>
      )}
    </div>
  );
}
const nodeTypes = { agent: GraphNode };

export function AgentGraph({ timeline, selected, onSelect }: { timeline: any[]; selected?: string | null; onSelect?: (node: string | null) => void }) {
  const { nodes, edges } = useMemo(() => {
    const path = ["start", ...timeline.map((e) => (e.node === "next" ? "next_job" : e.node)), "end"];
    const visits: Record<string, number> = {};
    const seconds: Record<string, number> = {};
    timeline.forEach((e) => {
      const n = e.node === "next" ? "next_job" : e.node;
      visits[n] = (visits[n] || 0) + 1;
      seconds[n] = (seconds[n] || 0) + (e.seconds || 0);
    });
    visits.start = visits.end = timeline.length ? 1 : 0;
    const traversed: Record<string, number> = {};
    for (let i = 0; i + 1 < path.length; i++) traversed[`${path[i]}>${path[i + 1]}`] = (traversed[`${path[i]}>${path[i + 1]}`] || 0) + 1;
    const nodes: Node[] = Object.entries(NODE_INFO).map(([id, info]) => ({
      id, type: "agent", position: { x: info.x, y: info.y }, draggable: false,
      data: { ...info, kind: id, visits: visits[id] || 0, seconds: seconds[id], selected: selected === id, onClick: () => onSelect?.(selected === id ? null : id) },
    }));
    const edges: Edge[] = EDGES.map(([s, t, sh, th, label]) => {
      const count = traversed[`${s}>${t}`] || 0;
      return {
        id: `${s}-${t}`, source: s, target: t, sourceHandle: sh, targetHandle: th, type: "smoothstep", animated: count > 0,
        label: count > 1 ? `×${count}${label ? ` · ${label}` : ""}` : count === 1 && label ? label : undefined,
        labelStyle: { fontSize: 10, fontWeight: 600 }, labelBgPadding: [4, 2] as [number, number], labelBgBorderRadius: 6,
        style: { stroke: count ? "#818cf8" : "#94a3b8", strokeWidth: count ? 2.2 : 1.2, strokeDasharray: count ? undefined : "4 4", opacity: count ? 1 : 0.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: count ? "#818cf8" : "#94a3b8" },
      };
    });
    return { nodes, edges };
  }, [timeline, selected, onSelect]);
  return (
    <div className="h-[380px] overflow-hidden rounded-2xl border hairline bg-gradient-to-br from-indigo-500/[0.04] to-transparent">
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.15 }} nodesConnectable={false} elementsSelectable={false} proOptions={{ hideAttribution: true }} minZoom={0.3}>
        <Background gap={22} size={1} color="rgb(148 163 184 / 0.25)" />
        <Controls showInteractive={false} position="bottom-left" />
      </ReactFlow>
    </div>
  );
}

// ------------------------------------------------------------------ timeline

function eventText(e: any): string {
  switch (e.node) {
    case "understand": return `${e.task === "one_job" ? "One job" : "Find the open jobs that fit"}: ${(e.jobs || []).join(", ")} — ${e.why || ""}`;
    case "plan": return `Round ${e.round} by ${e.by}: ${e.calls} tool call(s)${e.dropped ? `, ${e.dropped} refused by code` : ""}${e.thought ? ` — “${e.thought}”` : e.why ? ` — ${e.why}` : ""}`;
    case "act": return `${e.cv_searches} CV search(es), ${e.hr_searches} HR knowledge search(es) · ${e.hr_excerpts} HR excerpts so far`;
    case "evaluate": return `Weak: ${(e.weak || []).join(", ") || "none"} → ${e.next} (${e.why})`;
    case "analyse": return e.error ? `Model failed (${e.error}) → fast semantic estimate` : `Qwen judged every requirement (${e.prompt_chars || "?"} prompt characters)`;
    case "verify": return `${e.changes} claim(s) changed by the checks → grade ${e.grade}, score ${e.score}`;
    case "next": return e.next === "end" ? `Done: ${e.reports} report(s)` : `Next job: ${e.job}`;
    default: return e.why || "";
  }
}

export function AgentTimeline({ timeline, filter }: { timeline: any[]; filter?: string | null }) {
  const rows = timeline.map((e, i) => ({ e, i })).filter(({ e }) => !filter || (e.node === "next" ? "next_job" : e.node) === filter);
  return (
    <ol className="relative space-y-2 border-l hairline pl-5">
      {rows.map(({ e, i }) => (
        <li key={i} className="relative">
          <span className={cx("absolute -left-[27px] top-2 grid h-3.5 w-3.5 place-items-center rounded-full border-2 bg-[color:var(--bg)]", e.error ? "border-rose-500" : "border-indigo-400")} />
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="muted font-mono">#{i + 1}</span>
            <Badge tone={e.error ? "rose" : "indigo"}>{e.node}</Badge>
            {e.job && e.node !== "next" && <span className="muted">{e.job}</span>}
            {e.seconds !== undefined && <span className="muted">{num(e.seconds, 2)} s</span>}
            {e.prompt_tokens && <Badge tone="slate">{e.prompt_tokens} → {e.output_tokens} tokens</Badge>}
          </div>
          <div className="mt-0.5 text-[13px] leading-relaxed">{eventText(e)}</div>
        </li>
      ))}
    </ol>
  );
}

// ------------------------------------------------------------------ tool calls

export const TOOL_META: Record<string, { label: string; tone: string; icon: ReactNode; what: string }> = {
  search_cv: { label: "CV Search", tone: "cyan", icon: <FileSearch className="h-3.5 w-3.5" />, what: "BGE-M3 over the candidate's CV passages, reordered by the BGE reranker" },
  search_hr_knowledge: { label: "HR Knowledge Search", tone: "violet", icon: <Search className="h-3.5 w-3.5" />, what: "RAG: BGE-M3 in the Qdrant collection hr_knowledge, reordered by the BGE reranker" },
  semantic_matching: { label: "Semantic Matching", tone: "amber", icon: <Gauge className="h-3.5 w-3.5" />, what: "Fast grade of every open job (no LLM)" },
};

export function ToolCall({ call }: { call: any }) {
  const [open, setOpen] = useState(false);
  const meta = TOOL_META[call.tool] || { label: call.tool, tone: "slate", icon: <Wrench className="h-3.5 w-3.5" />, what: "" };
  const results: any[] = call.results || [];
  return (
    <div className="rounded-xl border hairline">
      <button onClick={() => setOpen(!open)} className="flex w-full items-start gap-3 px-3 py-2.5 text-left">
        <ChevronRight className={cx("mt-0.5 h-4 w-4 shrink-0 transition", open && "rotate-90")} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge tone={meta.tone}>{meta.icon}{meta.label}</Badge>
            {call.requirement && <Badge tone="indigo">{call.requirement}</Badge>}
            <span className="muted">round {call.round}</span>
            <span className="muted">· {call.found} found</span>
            {call.best !== null && call.best !== undefined && <span className="muted">· best {Number(call.best).toFixed(3)}</span>}
            {(call.documents || []).map((d: string) => <DocLink key={d} docId={d}>{d}</DocLink>)}
          </div>
          <div className="mt-1 truncate font-mono text-[12px]">{call.tool}(“{call.query || call.result}”)</div>
        </div>
      </button>
      {open && (
        <div className="space-y-2 border-t hairline px-3 py-3">
          <div className="muted text-[11px]">{meta.what}</div>
          {call.tool === "semantic_matching" && <div className="text-xs">{call.result}</div>}
          {!results.length && call.tool !== "semantic_matching" && <div className="muted text-xs">No result was kept in this trace (older run) or nothing was found.</div>}
          {results.map((r, i) => (
            <div key={i} className="rounded-lg surface-2 p-2.5 text-xs">
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <span className="font-mono text-[10px] muted">#{i + 1}</span>
                {r.doc_id ? <DocLink docId={r.doc_id} section={r.section} chunkId={r.chunk_id}>{r.doc_id}</DocLink> : <Badge tone="cyan">{r.label}</Badge>}
                {r.doc_id && <span className="muted">{r.section} · chunk {r.chunk_id}</span>}
                {!r.doc_id && <span className="muted">section: {r.section}</span>}
                <span className="ml-auto flex items-center gap-1.5"><span className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-500/20"><span className="block h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400" style={{ width: `${Math.max(4, Math.min(100, (r.score || 0) * 100))}%` }} /></span><span className="font-mono">{Number(r.score).toFixed(3)}</span></span>
              </div>
              <div className="leading-relaxed">{r.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ToolCalls({ calls }: { calls: any[] }) {
  const [tool, setTool] = useState<string>("all");
  const counts = calls.reduce((acc: Record<string, number>, c) => ({ ...acc, [c.tool]: (acc[c.tool] || 0) + 1 }), {});
  const shown = calls.filter((c) => tool === "all" || c.tool === tool);
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        <button onClick={() => setTool("all")} className={cx("chip", tool === "all" && "border-indigo-400 text-indigo-600 dark:text-indigo-300")}>All · {calls.length}</button>
        {Object.entries(counts).map(([t, n]) => (
          <button key={t} onClick={() => setTool(t)} className={cx("chip", tool === t && "border-indigo-400 text-indigo-600 dark:text-indigo-300")}>
            {TOOL_META[t]?.icon}{TOOL_META[t]?.label || t} · {n}
          </button>
        ))}
      </div>
      <div className="space-y-2">{shown.map((c, i) => <ToolCall key={i} call={c} />)}</div>
    </div>
  );
}

// ------------------------------------------------------------------ per job: plans, evidence, checks, prompt

export function AgentJobTrace({ job, systemPrompts }: { job: any; systemPrompts?: Record<string, string> }) {
  const agent = job.agent || {};
  const changes = job.checks?.verification || [];
  const evidence: Record<string, any[]> = agent.evidence || {};
  return (
    <div className="space-y-4">
      <KeyVals cols={4} items={[
        ["Planning rounds by Qwen", agent.planning_rounds ?? 0],
        ["Tool calls", (agent.tool_calls || []).length],
        ["Qwen judgement", job.llm_seconds ? `${job.llm_seconds} s · ${job.prompt_tokens} → ${job.output_tokens} tokens` : job.error ? "failed" : "—"],
        ["Claims changed by code", changes.length],
      ]} />
      {job.error && <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-700 dark:text-rose-300">The model failed ({job.error}): the fast semantic estimate was used.</div>}

      {!!(agent.plans || []).length && (
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold"><Brain className="h-4 w-4 text-indigo-500" /> Planning rounds — what the agent decided</div>
          <div className="space-y-2">
            {agent.plans.map((p: any, i: number) => (
              <div key={i} className="rounded-xl border hairline p-3 text-xs">
                <div className="flex flex-wrap items-center gap-2"><Badge tone="indigo">round {p.round}</Badge><Badge tone={p.by === "model" ? "violet" : "slate"}>by {p.by}</Badge>{p.weak?.length ? <span className="muted">weak: {p.weak.join(", ")}</span> : null}</div>
                {p.thought && <div className="mt-1.5 italic">“{p.thought}”</div>}
                {!!(p.calls || []).length && <div className="mt-2 space-y-1">{p.calls.map((c: any, j: number) => <div key={j} className="font-mono text-[11px]"><span className="text-emerald-600 dark:text-emerald-300">✓</span> {c.tool}({c.requirement ? `${c.requirement}: ` : ""}“{c.query}”)</div>)}</div>}
                {!!(p.dropped || []).length && <div className="mt-1 space-y-1">{p.dropped.map((c: any, j: number) => <div key={j} className="font-mono text-[11px] text-rose-600 dark:text-rose-300">✗ {c.tool}(“{c.query}”) — refused: {c.why}</div>)}</div>}
              </div>
            ))}
          </div>
        </div>
      )}

      <Collapsible title={<><Wrench className="h-4 w-4 text-cyan-500" /> Every tool call for this job, with what it returned ({(agent.tool_calls || []).length})</>}>
        <ToolCalls calls={agent.tool_calls || []} />
      </Collapsible>

      <Collapsible title={<><FileSearch className="h-4 w-4 text-cyan-500" /> CV evidence kept per requirement (reranked, best first)</>}>
        <div className="space-y-3">
          {Object.entries(evidence).map(([rid, hits]) => (
            <div key={rid}>
              <Badge tone="indigo">{rid}</Badge>
              <div className="mt-1.5 space-y-1.5">
                {hits.map((h: any, i: number) => (
                  <div key={i} className="flex gap-3 rounded-lg surface-2 p-2 text-xs">
                    <span className="w-14 shrink-0 font-mono">{Number(h.score).toFixed(3)}</span>
                    <span className="muted w-40 shrink-0 truncate">{h.label}</span>
                    <span className="leading-relaxed">{h.text}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {!Object.keys(evidence).length && <div className="muted text-sm">No evidence kept.</div>}
        </div>
      </Collapsible>

      <Collapsible title={<><ListChecks className="h-4 w-4 text-emerald-500" /> Checks by code on Qwen's claims ({changes.length})</>}>
        {changes.length ? (
          <table className="table-x"><thead><tr><th>Req.</th><th>Qwen said</th><th>Kept</th><th>Why</th></tr></thead>
            <tbody>{changes.map((c: any, i: number) => <tr key={i}><td><Badge tone="indigo">{c.id}</Badge></td><td>{c.from}</td><td className="font-medium">{c.to}</td><td className="text-xs">{c.why}</td></tr>)}</tbody>
          </table>
        ) : <div className="muted flex items-center gap-2 text-sm"><CheckCircle2 className="h-4 w-4 text-emerald-500" /> Every claim passed the checks.</div>}
      </Collapsible>

      <Collapsible title={<><Sparkles className="h-4 w-4 text-violet-500" /> The exact prompt sent to Qwen and its raw answer</>}>
        <div className="space-y-3">
          {systemPrompts?.analysis && <CodeBlock title="system prompt (analysis)" code={systemPrompts.analysis} maxHeight="max-h-56" />}
          {job.prompt && <CodeBlock title={`user prompt · ${job.prompt_chars || job.prompt.length} characters`} code={job.prompt} />}
          {job.raw_answer && <CodeBlock title="Qwen's raw answer (JSON)" code={job.raw_answer} />}
          {systemPrompts?.plan && <CodeBlock title="system prompt (planning rounds)" code={systemPrompts.plan} maxHeight="max-h-56" />}
          {!job.prompt && <div className="muted text-sm">No prompt stored.</div>}
        </div>
      </Collapsible>
    </div>
  );
}
