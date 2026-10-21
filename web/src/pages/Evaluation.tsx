import { useQuery } from "@tanstack/react-query";
import { BarChart3, Bot, Database, FileSearch, FlaskConical, GitCompareArrows, Network, Target, Workflow } from "lucide-react";
import { ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AssistantEval, ExtractionEval, FinetuneEval, MatchingEval, MlflowCard, PipelineStats, RagEval } from "../components/domain/EvalCharts";
import { Badge, Card, Loading, PageHeader, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { num, pct, pts } from "../lib/format";

type Tab = "overview" | "extraction" | "finetune" | "rag" | "matching" | "screening" | "assistant" | "pipeline";
const ICONS: Record<string, ReactNode> = { extraction: <FileSearch className="h-4 w-4" />, finetune: <FlaskConical className="h-4 w-4" />, rag: <Network className="h-4 w-4" />, matching: <Target className="h-4 w-4" />, screening: <Workflow className="h-4 w-4" />, assistant: <Bot className="h-4 w-4" /> };

function headline(key: string, h: any): ReactNode {
  if (!h) return <span className="muted">not run yet</span>;
  switch (key) {
    case "extraction": return <><b className="text-2xl">{pct(h.after, 1)}</b> <span className="muted text-xs">overall F1 · {pts(h.delta)} vs base</span></>;
    case "finetune": return <><b className="text-2xl">{num(h.last_loss, 3)}</b> <span className="muted text-xs">final loss (from {num(h.first_loss, 3)}) · {h.steps} steps</span></>;
    case "rag": { const r: any = h["dense+rerank"] || Object.values(h)[0] || {}; return <><b className="text-2xl">{pct(r["hit@1"])}</b> <span className="muted text-xs">Hit@1 with the reranker · MRR {num(r.mrr, 3)}</span></>; }
    case "matching": { const best = Object.entries(h).sort((a: any, b: any) => (b[1] ?? 0) - (a[1] ?? 0))[0] as any; return <><b className="text-2xl">{pct(best?.[1])}</b> <span className="muted text-xs">best test grade accuracy · {best?.[0]}</span></>; }
    case "screening": return <><b className="text-2xl">{h.mean_tool_calls}</b> <span className="muted text-xs">tool calls / pair · {h.checks_applied} claims checked</span></>;
    case "assistant": return <><b className="text-2xl">{pct(h.success)}</b> <span className="muted text-xs">task success · {pct(h.tool_choice)} right first tool</span></>;
    default: return null;
  }
}

export default function Evaluation() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const tab = (params.get("tab") as Tab) || "overview";
  const setTab = (t: Tab) => setParams({ tab: t });
  const overview = useQuery({ queryKey: ["eval-overview"], queryFn: () => api.get("/evaluation/overview") });
  const q = (key: string, enabled: boolean) => useQuery({ queryKey: [`eval-${key}`], queryFn: () => api.get(`/evaluation/${key}`), enabled });
  const extraction = q("extraction", tab === "extraction");
  const finetune = q("finetune", tab === "finetune");
  const rag = q("rag", tab === "rag");
  const matching = q("matching", tab === "matching" || tab === "screening");
  const assistant = q("assistant", tab === "assistant");
  const pipeline = q("pipeline", tab === "pipeline");
  const o = overview.data;
  return (
    <div>
      <PageHeader eyebrow="DSO1 → DSO4 · every technology measured" title="Evaluation & experiment tracking" icon={<BarChart3 className="h-7 w-7" />}
        subtitle="Every number comes from a run on Kaggle's GPUs on the frozen evaluation set eval_v1 (dev for tuning, test for the final numbers) — no model runs on our PCs — plus the live pipeline. The same results are logged in MLflow." />
      <Tabs value={tab} onChange={setTab} tabs={[
        { id: "overview", label: "Overview", icon: <BarChart3 className="h-4 w-4" /> }, { id: "extraction", label: "Extraction", icon: ICONS.extraction },
        { id: "finetune", label: "Fine-tuning", icon: ICONS.finetune }, { id: "rag", label: "RAG", icon: ICONS.rag }, { id: "matching", label: "Matching", icon: ICONS.matching },
        { id: "screening", label: "Screening agent", icon: ICONS.screening }, { id: "assistant", label: "Assistant", icon: ICONS.assistant }, { id: "pipeline", label: "Live pipeline", icon: <Database className="h-4 w-4" /> },
      ]} />
      {tab === "overview" && (
        <div className="space-y-5">
          <MlflowCard />
          {overview.isLoading ? <Loading rows={6} /> : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {(o?.components || []).map((c: any) => (
                <Card key={c.key} hover onClick={() => setTab(c.key)} className="flex flex-col p-5">
                  <div className="flex items-center justify-between gap-2"><Badge tone="indigo">{c.dso}</Badge><span className="text-indigo-500">{ICONS[c.key]}</span></div>
                  <div className="mt-3 font-display text-base font-semibold">{c.title}</div>
                  <div className="mt-3 font-display">{headline(c.key, o?.headline?.[c.key])}</div>
                  <div className="muted mt-3 text-xs leading-relaxed"><b className="text-[color:var(--text)]">Measures:</b> {c.metrics}</div>
                  <div className="muted mt-1.5 text-xs leading-relaxed"><b className="text-[color:var(--text)]">Data:</b> {c.data}</div>
                  <div className="mt-auto flex flex-wrap gap-1 pt-3">{c.technologies.map((t: string) => <Badge key={t} tone="slate">{t}</Badge>)}</div>
                </Card>
              ))}
              <Card hover onClick={() => setTab("pipeline")} className="flex flex-col p-5">
                <div className="flex items-center justify-between gap-2"><Badge tone="cyan">LIVE</Badge><Database className="h-4 w-4 text-cyan-500" /></div>
                <div className="mt-3 font-display text-base font-semibold">Application pipeline (live)</div>
                <div className="muted mt-3 text-xs leading-relaxed">Timings and outcomes of the 12 steps over the real applications: document processing, OCR, extraction, indexing, screening agent, e-mails, decisions.</div>
                <div className="mt-auto flex flex-wrap gap-1 pt-3">{["PostgreSQL", "Qdrant", "n8n", "SMTP / Outbox", "Google Meet / Jitsi"].map((t) => <Badge key={t} tone="slate">{t}</Badge>)}</div>
              </Card>
              <Card hover onClick={() => navigate("/models")} className="flex flex-col justify-center p-5">
                <GitCompareArrows className="h-6 w-6 text-indigo-500" />
                <div className="mt-3 font-display text-base font-semibold">Base vs fine-tuned, live</div>
                <div className="muted mt-1 text-xs">Run both models on any CV and compare their profiles field by field.</div>
              </Card>
            </div>
          )}
        </div>
      )}
      {tab === "extraction" && (extraction.isLoading ? <Loading rows={6} /> : <ExtractionEval data={extraction.data} />)}
      {tab === "finetune" && (finetune.isLoading ? <Loading rows={6} /> : <FinetuneEval data={finetune.data} />)}
      {tab === "rag" && (rag.isLoading ? <Loading rows={6} /> : <RagEval data={rag.data} />)}
      {tab === "matching" && (matching.isLoading ? <Loading rows={6} /> : <MatchingEval data={matching.data} />)}
      {tab === "screening" && (matching.isLoading ? <Loading rows={6} /> : <MatchingEval data={matching.data} focusAgent />)}
      {tab === "assistant" && (assistant.isLoading ? <Loading rows={6} /> : <AssistantEval data={assistant.data} />)}
      {tab === "pipeline" && (pipeline.isLoading ? <Loading rows={6} /> : <PipelineStats data={pipeline.data} />)}
    </div>
  );
}
