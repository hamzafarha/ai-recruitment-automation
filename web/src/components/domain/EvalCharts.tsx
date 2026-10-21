// Charts of the evaluations (data/evaluation/, runs on Kaggle): extraction before/after fine-tuning, the
// QLoRA loss curve, RAG retrieval, matching and the screening agent, the assistant, the live pipeline.
import { useQuery } from "@tanstack/react-query";
import { Activity, Database, ExternalLink, FlaskConical, Gauge, TrendingDown, TrendingUp } from "lucide-react";
import { useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../../lib/api";
import { AFTER, axisProps, BEFORE, PALETTE, tooltipStyle } from "../../lib/chart";
import { num, pct, pts, secs } from "../../lib/format";
import { Badge, Callout, Card, CardHeader, CodeBlock, Collapsible, cx, Empty, KeyVals, Markdown, Stat, Tabs } from "../ui";

const pctTick = (v: number) => `${Math.round(v)}%`;

export function Missing({ job, what }: { job: string; what: string }) {
  return (
    <Card><Empty icon={<FlaskConical className="h-6 w-6" />} title={`No ${what} results on this computer yet`}>
      Run it on Kaggle's GPUs, then download it: <code>python scripts/kaggle_run.py all {job}</code> (or <code>fetch {job}</code> if it already ran).
    </Empty></Card>
  );
}

// ------------------------------------------------------------------ DSO1 extraction

export function ExtractionEval({ data }: { data: any }) {
  const [group, setGroup] = useState<"input_type" | "language" | "source">("input_type");
  if (!data?.available) return <Missing job="extraction" what="extraction" />;
  if (!data.metrics) return <Callout tone="amber">Only one extraction run: the fine-tuned run appears after <code>python scripts/kaggle_run.py all extraction-ft</code>.</Callout>;
  const overall = data.metrics.find((m: any) => m.metric === "Overall F1");
  const bars = data.metrics.map((m: any) => ({ metric: m.metric, Base: (m.before ?? 0) * 100, "Fine-tuned": (m.after ?? 0) * 100 }));
  const radar = data.sections.map((s: any) => ({ section: s.section, Base: (s.before_f1 ?? 0) * 100, "Fine-tuned": (s.after_f1 ?? 0) * 100 }));
  const groups = (data.groups[group] || []).map((g: any) => ({ name: `${g[group]} (${g.documents})`, Base: (g.before ?? 0) * 100, "Fine-tuned": (g.after ?? 0) * 100 }));
  const docs: any[] = data.documents || [];
  const better = docs.filter((d) => (d.delta ?? 0) > 0.005).length;
  const worse = docs.filter((d) => (d.delta ?? 0) < -0.005).length;
  const hb = data.identity?.before?.hallucinations, ha = data.identity?.after?.hallucinations;
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Overall F1 — fine-tuned" value={pct(overall?.after, 1)} hint={<span className={(overall?.delta ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-300" : "text-rose-500"}>{pts(overall?.delta)} vs base ({pct(overall?.before, 1)})</span>} icon={<TrendingUp className="h-4 w-4" />} tone="indigo" />
        <Stat label="Identity hallucinations" value={ha ?? "—"} hint={`base: ${hb ?? "—"} invented name/e-mail/phone fields`} icon={<Gauge className="h-4 w-4" />} tone={ha !== undefined && hb !== undefined && ha <= hb ? "emerald" : "amber"} />
        <Stat label="Seconds per CV" value={num(data.speed?.after, 1)} hint={`base: ${num(data.speed?.before, 1)} s (2× T4, fp16)`} icon={<Activity className="h-4 w-4" />} tone="cyan" />
        <Stat label="CVs improved / worse" value={`${better} / ${worse}`} hint={`${docs.length} CVs of the frozen dev split, same code`} icon={<TrendingDown className="h-4 w-4" />} tone="violet" />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card>
          <CardHeader title="Every metric, base vs fine-tuned" subtitle="0-100 % axis (a truncated axis would exaggerate small gaps)" />
          <div className="h-80 px-2 pb-3">
            <ResponsiveContainer>
              <BarChart data={bars} layout="vertical" margin={{ top: 10, right: 20, left: 40, bottom: 0 }}>
                <CartesianGrid horizontal={false} strokeOpacity={0.1} />
                <XAxis type="number" domain={[0, 100]} tickFormatter={pctTick} {...axisProps} />
                <YAxis type="category" dataKey="metric" width={110} {...axisProps} />
                <Tooltip {...tooltipStyle} formatter={(v: any) => `${Number(v).toFixed(1)} %`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Base" fill={BEFORE} radius={[0, 4, 4, 0]} />
                <Bar dataKey="Fine-tuned" fill={AFTER} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader title="F1 per CV section" subtitle="Positions, degrees, skills" />
          <div className="h-80">
            <ResponsiveContainer>
              <RadarChart data={radar} outerRadius="72%">
                <PolarGrid strokeOpacity={0.2} />
                <PolarAngleAxis dataKey="section" tick={{ fontSize: 12, fill: "currentColor" }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9, fill: "currentColor", fillOpacity: 0.5 }} />
                <Radar name="Base" dataKey="Base" stroke={BEFORE} fill={BEFORE} fillOpacity={0.15} />
                <Radar name="Fine-tuned" dataKey="Fine-tuned" stroke={AFTER} fill={AFTER} fillOpacity={0.3} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Tooltip {...tooltipStyle} formatter={(v: any) => `${Number(v).toFixed(1)} %`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <Card>
        <CardHeader title="Overall F1 by document type, language and source" subtitle="Scanned PDFs go through OCR (PaddleOCR models) before Qwen" actions={<Tabs className="mb-0" value={group} onChange={setGroup} tabs={[{ id: "input_type", label: "Input type" }, { id: "language", label: "Language" }, { id: "source", label: "Source" }]} />} />
        <div className="h-64 px-2 pb-3">
          <ResponsiveContainer>
            <BarChart data={groups} margin={{ top: 16, right: 16, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.1} />
              <XAxis dataKey="name" {...axisProps} />
              <YAxis domain={[0, 100]} tickFormatter={pctTick} {...axisProps} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => `${Number(v).toFixed(1)} %`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Base" fill={BEFORE} radius={[6, 6, 0, 0]} />
              <Bar dataKey="Fine-tuned" fill={AFTER} radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Collapsible title={<>Per CV — {better} improved, {worse} worse, {docs.length - better - worse} unchanged</>}>
        <div className="max-h-96 overflow-y-auto">
          <table className="table-x"><thead><tr><th>CV</th><th>Type</th><th>Lang.</th><th>Base</th><th>Fine-tuned</th><th>Change</th></tr></thead>
            <tbody>{docs.map((d) => (
              <tr key={d.doc_id}><td className="font-mono text-xs">{d.doc_id}</td><td>{d.input_type}</td><td>{d.language}</td><td>{pct(d.before)}</td><td>{pct(d.after)}</td>
                <td><span className={cx("font-semibold", (d.delta ?? 0) > 0.005 ? "text-emerald-600 dark:text-emerald-300" : (d.delta ?? 0) < -0.005 ? "text-rose-500" : "muted")}>{pts(d.delta)}</span>{d.error_before && <Badge tone="rose" className="ml-1">error before</Badge>}{d.error_after && <Badge tone="rose" className="ml-1">error after</Badge>}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Collapsible>
      <Collapsible title="Run configurations">
        <div className="grid gap-3 md:grid-cols-2">{(["before", "after"] as const).map((k) => <CodeBlock key={k} title={k === "before" ? `base · ${data.before}` : `fine-tuned · ${data.after}`} code={JSON.stringify(data.config?.[k], null, 2)} maxHeight="max-h-60" />)}</div>
      </Collapsible>
    </div>
  );
}

export function FinetuneEval({ data }: { data: any }) {
  if (!data?.available) return <Missing job="finetune" what="fine-tuning" />;
  const losses = (data.losses || []).map((l: any) => ({ step: l.step, loss: l.loss }));
  const p = data.params || {};
  const dr = data.data_report || {};
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Training loss" value={`${num(losses[0]?.loss, 3)} → ${num(losses.at(-1)?.loss, 3)}`} hint={`${data.global_step}/${data.max_steps} steps · ${data.status}`} icon={<TrendingDown className="h-4 w-4" />} tone="indigo" />
        <Stat label="Training CVs" value={data.examples ?? "—"} hint={`${data.skipped_too_long ?? 0} skipped (too long for ${p.MAX_SEQ_LEN} tokens)`} icon={<Database className="h-4 w-4" />} tone="cyan" />
        <Stat label="LoRA" value={`r=${p.LORA_R} · α=${p.LORA_ALPHA}`} hint={`lr ${p.LEARNING_RATE} · dropout ${p.LORA_DROPOUT} · ${p.EPOCHS} epoch`} icon={<FlaskConical className="h-4 w-4" />} tone="violet" />
        <Stat label="Training time" value={secs((data.minutes || 0) * 60)} hint={`${data.gpu_processes || 2}× Tesla T4 (Kaggle)`} icon={<Activity className="h-4 w-4" />} tone="emerald" />
      </div>
      <Card>
        <CardHeader title="Loss curve (QLoRA, 4-bit Qwen2.5-7B-Instruct)" subtitle="Lower = the model reproduces the reference JSON better" />
        <div className="h-72 px-2 pb-3">
          <ResponsiveContainer>
            <AreaChart data={losses} margin={{ top: 16, right: 20, left: -10, bottom: 0 }}>
              <defs><linearGradient id="loss" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={AFTER} stopOpacity={0.45} /><stop offset="100%" stopColor={AFTER} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeOpacity={0.1} />
              <XAxis dataKey="step" {...axisProps} />
              <YAxis {...axisProps} domain={["auto", "auto"]} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => Number(v).toFixed(4)} />
              <Area type="monotone" dataKey="loss" stroke={AFTER} strokeWidth={2.5} fill="url(#loss)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="p-5">
          <div className="mb-3 text-sm font-semibold">Training data</div>
          <KeyVals cols={2} items={[["Candidate CVs", dr.candidates], ["Kept for training", dr.train], ["Held out", dr.holdout], ["Removed: poorly grounded", dr.removed_poorly_grounded], ["Removed: in the evaluation splits", dr.removed_in_eval_splits], ["Removed: repeated text", dr.removed_repeated_text], ["Labels", dr.labels]]} />
        </Card>
        <Card className="p-5">
          <div className="mb-3 text-sm font-semibold">Quick check on held-out CVs (smoke test)</div>
          {data.smoke_test ? <KeyVals cols={2} items={[["CVs", data.smoke_test.cvs], ["Errors", data.smoke_test.errors], ["Experience F1", pct(data.smoke_test.experience_f1)], ["Education F1", pct(data.smoke_test.education_f1)], ["Skills F1", pct(data.smoke_test.skills_f1)], ["Overall F1", pct(data.smoke_test.overall_f1)]]} /> : <span className="muted">—</span>}
          {data.environment && <div className="muted mt-4 text-xs">torch {data.environment.torch} · transformers {data.environment.transformers} · peft {data.environment.peft} · bitsandbytes {data.environment.bitsandbytes}</div>}
        </Card>
      </div>
      {data.report && <Collapsible title="Fine-tuning report"><Markdown>{data.report}</Markdown></Collapsible>}
    </div>
  );
}

// ------------------------------------------------------------------ DSO2 retrieval

export function RagEval({ data }: { data: any }) {
  const [slice, setSlice] = useState("all");
  if (!data?.available) return <Missing job="rag" what="retrieval (RAG)" />;
  const configs: string[] = data.configurations;
  const slices = Object.keys(data.metrics[configs[0]] || {});
  const rows = ["hit@1", "hit@3", "hit@5", "mrr", "recall@1", "recall@3", "recall@5"].map((m) => ({ metric: m.toUpperCase().replace("HIT", "Hit").replace("RECALL", "Recall"), ...Object.fromEntries(configs.map((c) => [c, (data.metrics[c]?.[slice]?.[m] ?? 0) * 100])) }));
  const best = configs.find((c) => c === "dense+rerank") || configs[0];
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {configs.map((c, i) => <Stat key={c} label={c} value={pct(data.metrics[c]?.all?.["hit@1"])} hint={`Hit@1 · MRR ${num(data.metrics[c]?.all?.mrr, 3)} · Hit@3 ${pct(data.metrics[c]?.all?.["hit@3"])}`} tone={["slate", "indigo", "cyan", "violet"][i % 4]} icon={<Database className="h-4 w-4" />} />)}
      </div>
      <Card>
        <CardHeader title="Retrieval quality per configuration" subtitle={`${data.metrics[best]?.[slice]?.queries ?? "?"} questions · Hit@k = the expected document is in the top k · MRR = 1 / rank of the first expected document`}
          actions={<select className="input w-auto" value={slice} onChange={(e) => setSlice(e.target.value)}>{slices.map((s) => <option key={s} value={s}>{s}</option>)}</select>} />
        <div className="h-80 px-2 pb-3">
          <ResponsiveContainer>
            <BarChart data={rows} margin={{ top: 16, right: 16, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.1} />
              <XAxis dataKey="metric" {...axisProps} />
              <YAxis domain={[0, 100]} tickFormatter={pctTick} {...axisProps} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => `${Number(v).toFixed(1)} %`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {configs.map((c, i) => <Bar key={c} dataKey={c} fill={PALETTE[i % PALETTE.length]} radius={[4, 4, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Collapsible title={`Per question (${(data.per_query || []).length} rows)`}>
        <div className="max-h-96 overflow-y-auto">
          <table className="table-x"><thead><tr><th>Config</th><th>Q</th><th>Lang.</th><th>Cross-lingual</th><th>Hit@1</th><th>Hit@3</th><th>RR</th><th>Top 3 documents</th></tr></thead>
            <tbody>{(data.per_query || []).map((r: any, i: number) => <tr key={i}><td className="text-xs">{r.configuration}</td><td className="font-mono text-xs">{r.id}</td><td>{r.language}</td><td>{r.cross_lingual === "True" ? "yes" : ""}</td><td>{r["hit@1"] === "1.0" ? "✓" : "·"}</td><td>{r["hit@3"] === "1.0" ? "✓" : "·"}</td><td>{num(Number(r.rr), 2)}</td><td className="font-mono text-[11px]">{r.top3}</td></tr>)}</tbody>
          </table>
        </div>
      </Collapsible>
      {data.report && <Collapsible title="Full report"><Markdown>{data.report}</Markdown></Collapsible>}
    </div>
  );
}

// ------------------------------------------------------------------ DSO3-4 matching & screening agent

export function MatchingEval({ data, focusAgent }: { data: any; focusAgent?: boolean }) {
  const [split, setSplit] = useState<"test" | "dev" | "all">("test");
  if (!data?.available) return <Missing job="matching" what="matching" />;
  const systems = Object.entries(data.systems || {}).filter(([name]) => !focusAgent || /^(llm|agent|semantic \(calibrated)/.test(name));
  const rows = systems.map(([name, s]: [string, any]) => ({ name, g: s[split]?.grades || {}, r: s[split]?.requirements || {} })).filter((x) => x.g.pairs);
  const chart = [
    { metric: "Grade accuracy", ...Object.fromEntries(rows.map((x) => [x.name, (x.g.accuracy ?? 0) * 100])) },
    { metric: "Within one grade", ...Object.fromEntries(rows.map((x) => [x.name, (x.g.within_one ?? 0) * 100])) },
    { metric: "Must-have status acc.", ...Object.fromEntries(rows.map((x) => [x.name, (x.r.accuracy ?? 0) * 100])) },
    { metric: "Must-have macro-F1", ...Object.fromEntries(rows.map((x) => [x.name, (x.r.macro_f1 ?? 0) * 100])) },
  ];
  const agent = data.agent;
  return (
    <div className="space-y-5">
      {focusAgent && agent && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Qwen calls / pair" value={agent.mean_llm_calls} hint="planning rounds + final judgement" tone="indigo" icon={<Activity className="h-4 w-4" />} />
          <Stat label="Tool calls / pair" value={agent.mean_tool_calls} hint={`${agent.pairs_with_planning}/${agent.pairs} pairs with planning rounds`} tone="cyan" icon={<Gauge className="h-4 w-4" />} />
          <Stat label="Claims changed by checks" value={agent.checks_applied} hint={Object.entries(agent.checks_by_kind || {}).map(([k, v]) => `${k}: ${v}`).join(" · ")} tone="violet" icon={<FlaskConical className="h-4 w-4" />} />
          <Stat label="Fallbacks" value={agent.fallbacks} hint="model failures → fast semantic estimate" tone="amber" icon={<TrendingDown className="h-4 w-4" />} />
        </div>
      )}
      <Card>
        <CardHeader title={focusAgent ? "Screening agent vs the fixed analysis (same Qwen2.5-7B, same pairs)" : "Every matching system on the graded pairs"} subtitle={`${data.pairs} CV-job pairs, silver labels from the rubric KB-RR-002 · thresholds calibrated on dev, test = final numbers`}
          actions={<Tabs className="mb-0" value={split} onChange={setSplit} tabs={[{ id: "test", label: "test" }, { id: "dev", label: "dev" }, { id: "all", label: "all" }]} />} />
        <div className="h-80 px-2 pb-3">
          <ResponsiveContainer>
            <BarChart data={chart} margin={{ top: 16, right: 16, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.1} />
              <XAxis dataKey="metric" {...axisProps} />
              <YAxis domain={[0, 100]} tickFormatter={pctTick} {...axisProps} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => `${Number(v).toFixed(1)} %`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {rows.map((x, i) => <Bar key={x.name} dataKey={x.name} fill={PALETTE[i % PALETTE.length]} radius={[4, 4, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="overflow-x-auto px-3 pb-3">
          <table className="table-x"><thead><tr><th>System</th><th>Pairs</th><th>Grade acc.</th><th>Within one</th><th>MAE</th><th>Quadratic κ</th><th>Spearman</th><th>Must-have acc.</th><th>Macro-F1</th></tr></thead>
            <tbody>{rows.map((x) => <tr key={x.name}><td className="font-medium">{x.name}</td><td>{x.g.pairs}</td><td>{pct(x.g.accuracy)}</td><td>{pct(x.g.within_one)}</td><td>{num(x.g.mae)}</td><td>{num(x.g.qwk)}</td><td>{num(x.g.spearman)}</td><td>{pct(x.r.accuracy)}</td><td>{num(x.r.macro_f1)}</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
      {data.thresholds && !focusAgent && <div className="muted text-xs">Calibrated thresholds (used by the app): demonstrated ≥ {data.thresholds.demonstrated}, equivalent ≥ {data.thresholds.equivalent} ({data.thresholds.calibrated_on}).</div>}
      {(data.notes || []).map((n: string, i: number) => <Callout key={i} tone="amber">{n}</Callout>)}
      {!focusAgent && <Collapsible title={`Per pair (${(data.per_pair || []).length} rows)`}>
        <div className="max-h-96 overflow-y-auto"><table className="table-x"><thead><tr><th>System</th><th>Pair</th><th>Split</th><th>Role</th><th>CV</th><th>Reference</th><th>Predicted</th><th>Score</th></tr></thead>
          <tbody>{(data.per_pair || []).map((r: any, i: number) => <tr key={i}><td className="text-xs">{r.system}</td><td className="font-mono text-xs">{r.pair_id}</td><td>{r.split}</td><td className="text-xs">{r.job_role}</td><td className="font-mono text-[11px]">{r.cv_id}</td><td>{r.reference_grade}</td><td><span className={r.reference_grade === r.predicted_grade ? "text-emerald-600 dark:text-emerald-300" : "text-rose-500"}>{r.predicted_grade}</span></td><td>{r.score}</td></tr>)}</tbody>
        </table></div>
      </Collapsible>}
      {data.report && !focusAgent && <Collapsible title="Full report"><Markdown>{data.report}</Markdown></Collapsible>}
    </div>
  );
}

// ------------------------------------------------------------------ DSO4 assistant

export function AssistantEval({ data }: { data: any }) {
  if (!data?.available) return <Missing job="agent" what="AI assistant" />;
  const o = data.summary?.overall || {};
  const r = data.summary?.refusals || {};
  const cats = Object.entries(data.summary?.by_category || {}).map(([k, v]: [string, any]) => ({ category: k, Success: v.success * 100, "Tool choice": (v.tool_choice ?? 0) * 100, Answer: (v.answer ?? 0) * 100 }));
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <Stat label="Task success" value={pct(o.success)} hint={`${o.scenarios} questions`} tone="indigo" />
        <Stat label="Right first tool" value={pct(o.tool_choice)} tone="cyan" />
        <Stat label="Expected answer" value={pct(o.answer)} tone="violet" />
        <Stat label="Grounded names" value={pct(o.grounded)} hint="no invented candidate" tone="emerald" />
        <Stat label="Refusals" value={pct(r.success)} hint={`${r.scenarios} discriminatory requests`} tone="amber" />
      </div>
      <Card>
        <CardHeader title="Per category" subtitle={`${o.mean_tool_calls} tool calls and ${o.mean_llm_calls} Qwen calls per question · ${o.mean_seconds} s per question · ${data.model}`} />
        <div className="h-72 px-2 pb-3">
          <ResponsiveContainer>
            <BarChart data={cats} margin={{ top: 16, right: 16, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.1} />
              <XAxis dataKey="category" {...axisProps} />
              <YAxis domain={[0, 100]} tickFormatter={pctTick} {...axisProps} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => `${Number(v).toFixed(0)} %`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Success" fill={PALETTE[0]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="Tool choice" fill={PALETTE[1]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="Answer" fill={PALETTE[2]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Collapsible title={`Every scenario (${data.scenarios?.length || 0})`}>
        <div className="space-y-2">{(data.scenarios || []).map((s: any) => (
          <Collapsible key={s.id} title={<span className="flex flex-wrap items-center gap-2 text-xs"><Badge tone={s.score?.success ? "emerald" : "rose"}>{s.score?.success ? "success" : "failed"}</Badge><span className="font-mono">{s.id}</span><Badge tone="slate">{s.category}</Badge><span className="truncate">{s.question}</span></span>}>
            <div className="space-y-2 text-xs">
              <div>Expected tools: <b>{(s.expected_tools || []).join(", ") || "none"}</b> · called: <b>{(s.tools || []).join(" → ") || "none"}</b> · {num(s.seconds, 1)} s</div>
              <Markdown className="text-xs">{s.answer || "—"}</Markdown>
            </div>
          </Collapsible>
        ))}</div>
      </Collapsible>
      {data.report && <Collapsible title="Full report"><Markdown>{data.report}</Markdown></Collapsible>}
    </div>
  );
}

// ------------------------------------------------------------------ live pipeline, MLflow

export function PipelineStats({ data }: { data: any }) {
  if (!data) return null;
  const steps = (data.steps || []).map((s: any) => ({ ...s, ok: s.done, ko: s.failed }));
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Applications" value={data.applications} hint={Object.entries(data.by_status || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "—"} tone="indigo" />
        <Stat label="Interviews" value={data.decisions?.interview ?? 0} hint={`shortlisted ${data.decisions?.shortlist ?? 0} · rejected ${data.decisions?.reject ?? 0}`} tone="violet" />
        <Stat label="E-mails" value={Object.values(data.emails || {}).reduce((a: number, b: any) => a + Number(b), 0) as number} hint={Object.entries(data.emails || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "—"} tone="cyan" />
        <Stat label="Failed steps" value={steps.reduce((a: number, s: any) => a + s.failed, 0)} hint="recorded, then continued" tone="rose" />
      </div>
      <Card>
        <CardHeader title="Mean and max seconds per step" subtitle="The live system (PostgreSQL applications table), not a benchmark" />
        <div className="h-80 px-2 pb-3">
          <ResponsiveContainer>
            <BarChart data={steps} margin={{ top: 16, right: 16, left: -10, bottom: 30 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.1} />
              <XAxis dataKey="label" {...axisProps} angle={-25} textAnchor="end" interval={0} height={60} tick={{ ...axisProps.tick, fontSize: 10 }} />
              <YAxis {...axisProps} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => secs(Number(v))} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="mean_seconds" name="mean" fill={PALETTE[0]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_seconds" name="max" fill={PALETTE[2]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="overflow-x-auto px-3 pb-3">
          <table className="table-x"><thead><tr><th>Step</th><th>Part</th><th>Runs</th><th>Done</th><th>Failed</th><th>Waiting</th><th>Mean</th><th>Max</th></tr></thead>
            <tbody>{steps.map((s: any) => <tr key={s.step}><td className="font-medium">{s.label}</td><td>{s.part}</td><td>{s.runs}</td><td>{s.done}</td><td className={s.failed ? "text-rose-500" : ""}>{s.failed}</td><td>{s.waiting}</td><td>{secs(s.mean_seconds)}</td><td>{secs(s.max_seconds)}</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

export function MlflowCard() {
  const status = useQuery({ queryKey: ["mlflow"], queryFn: () => api.get("/evaluation/mlflow"), refetchInterval: 30_000 });
  const s = status.data || {};
  const cmd1 = "python scripts/mlflow_sync.py";
  const cmd2 = `python -m mlflow ui --backend-store-uri ${s.tracking_uri || "sqlite:///mlruns/mlflow.db"} --port 5000`;
  return (
    <Card className="overflow-hidden">
      <div className="grid md:grid-cols-[1fr_auto]">
        <div className="p-5">
          <div className="flex items-center gap-2 font-display font-semibold"><FlaskConical className="h-4 w-4 text-indigo-500" /> MLflow experiment tracking
            <Badge tone={s.reachable ? "emerald" : "slate"} dot>{s.reachable ? `running · ${s.experiments?.length || 0} experiments` : s.installed ? "installed, interface not running" : "not installed"}</Badge></div>
          <p className="muted mt-1.5 text-sm">Every evaluation below is also logged in MLflow — one experiment per part (extraction base vs QLoRA, fine-tuning loss curve, RAG, matching, screening agent, assistant), with params, metrics and the report files as artifacts, to compare runs side by side.</p>
          <div className="mt-3 grid gap-2 md:grid-cols-2"><CodeBlock title="1 · log the evaluations" code={cmd1} maxHeight="max-h-20" /><CodeBlock title="2 · open the interface" code={cmd2} maxHeight="max-h-20" /></div>
          {!!s.experiments?.length && <div className="mt-3 flex flex-wrap gap-2">{s.experiments.map((e: any) => <a key={e.id} href={e.link} target="_blank" rel="noreferrer" className="chip hover:border-indigo-400">{e.name}<ExternalLink className="h-3 w-3" /></a>)}</div>}
        </div>
        <div className="flex items-center justify-center border-t hairline bg-gradient-to-br from-indigo-500/10 to-cyan-500/10 p-5 md:border-l md:border-t-0">
          <a href={s.ui_url || "http://localhost:5000"} target="_blank" rel="noreferrer" className={cx("btn-primary", !s.reachable && "opacity-60")}><ExternalLink className="h-4 w-4" /> Open MLflow</a>
        </div>
      </div>
    </Card>
  );
}

