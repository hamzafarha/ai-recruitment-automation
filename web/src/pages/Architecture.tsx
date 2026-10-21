import { useQuery } from "@tanstack/react-query";
import { Background, Edge, Handle, MarkerType, Node, NodeProps, Position, ReactFlow } from "@xyflow/react";
import { Map as MapIcon } from "lucide-react";
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { ErrorState } from "../components/common";
import { Badge, Card, CardHeader, cx, KeyVals, Loading, PageHeader } from "../components/ui";
import { api } from "../lib/api";

function PartNode({ data }: NodeProps) {
  const d = data as any;
  return (
    <div onClick={d.onClick} className={cx("w-[230px] cursor-pointer rounded-2xl border bg-white p-3 text-xs shadow-sm transition hover:-translate-y-0.5 hover:border-indigo-400 dark:bg-[#121a31]", d.ok === false ? "border-rose-400/60" : "hairline")}>
      <Handle type="target" position={Position.Left} /><Handle type="source" position={Position.Right} />
      <Handle type="target" id="t" position={Position.Top} /><Handle type="source" id="b" position={Position.Bottom} />
      <div className="flex items-center gap-2"><span className="grid h-6 w-6 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-500 text-[11px] font-bold text-white">{d.part}</span><span className="font-display text-[13px] font-semibold">{d.title}</span></div>
      <div className="muted mt-1.5 leading-relaxed">{d.tech}</div>
      <div className="mt-2 flex flex-wrap gap-1">{(d.stats || []).map((s: string) => <span key={s} className="rounded-md bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-700 dark:text-indigo-300">{s}</span>)}</div>
    </div>
  );
}
const nodeTypes = { part: PartNode };

export default function Architecture() {
  const navigate = useNavigate();
  const o = useQuery({ queryKey: ["overview"], queryFn: () => api.get("/observe/overview"), refetchInterval: 20_000 });
  const graph = useMemo(() => {
    const d = o.data;
    if (!d) return { nodes: [], edges: [] };
    const t = d.tables || {};
    const v = d.vectors || {};
    const parts = [
      { id: "1", part: 1, title: "Intake", tech: "Candidate portal, recruiter upload, n8n webhook", stats: [`${t.applications ?? 0} applications`], to: "/portal", x: 0, y: 0 },
      { id: "2", part: 2, title: "n8n orchestrator", tech: "One execution per application, Wait node for the decision", stats: [d.n8n?.reachable ? "reachable" : "not reachable"], ok: d.n8n?.reachable, to: "/notifications", x: 280, y: -150 },
      { id: "3", part: 3, title: "Document processing", tech: "PyMuPDF · python-docx · RapidOCR (PaddleOCR)", stats: [d.last_extraction?.input_type || "—"], to: "/pipeline", x: 280, y: 0 },
      { id: "4", part: 4, title: "CV extraction", tech: "Qwen2.5-7B + QLoRA adapter (Transformers, Kaggle GPU)", stats: [`${t.candidates ?? 0} candidates`, d.ai_server?.adapter ? "adapter loaded" : d.models?.llm_provider], ok: d.ai_server?.used ? d.ai_server?.reachable : undefined, to: "/models", x: 560, y: 0 },
      { id: "8", part: 8, title: "Data storage", tech: "PostgreSQL (tables) + Qdrant (BGE-M3 vectors)", stats: [`${Object.values(t).reduce((a: number, b: any) => a + Number(b), 0)} rows`, `${typeof v === "object" ? Object.values(v).reduce((a: number, b: any) => a + (Number(b) || 0), 0) : 0} vectors`], to: "/observe/database", x: 840, y: 0 },
      { id: "6", part: 6, title: "Knowledge sources", tech: "15 HR documents, jobs, candidates in Qdrant", stats: [`${v.hr_knowledge ?? 0} HR chunks`], to: "/knowledge", x: 840, y: 170 },
      { id: "5", part: 5, title: "Screening agent (RAG)", tech: "LangGraph · BGE-M3 · BGE reranker · Qwen judgement · code checks", stats: [`${t.analyses ?? 0} analyses`, `${t.agent_runs ?? 0} assistant runs`], to: "/observe/matching", x: 560, y: 170 },
      { id: "7", part: 7, title: "Notifications & e-mails", tech: "SMTP / outbox, Google Meet or Jitsi, calendar invites", stats: [`${t.emails ?? 0} e-mails`, `${t.notifications ?? 0} notifications`], to: "/outbox", x: 280, y: 170 },
      { id: "9", part: 9, title: "Human validation", tech: "A recruiter decides (KB-RR-001), the decision resumes the workflow", stats: [`${t.reviews ?? 0} decisions`], to: "/validation", x: 0, y: 170 },
    ];
    const nodes: Node[] = parts.map((p) => ({ id: p.id, type: "part", position: { x: p.x, y: p.y }, data: { ...p, onClick: () => navigate(p.to) }, draggable: false }));
    const link = (s: string, t2: string, label?: string, sh?: string, th?: string): Edge => ({ id: `${s}-${t2}`, source: s, target: t2, sourceHandle: sh, targetHandle: th, animated: true, type: "smoothstep", label, labelStyle: { fontSize: 10 }, style: { stroke: "#818cf8", strokeWidth: 1.8 }, markerEnd: { type: MarkerType.ArrowClosed, color: "#818cf8" } });
    const edges = [link("1", "3"), link("3", "4"), link("4", "8"), link("8", "6", undefined, "b", "t"), link("6", "5", "RAG"), link("5", "7", "report"), link("7", "9", "review needed"), link("2", "3", "orchestrates", "b", "t")];
    return { nodes, edges };
  }, [o.data, navigate]);
  if (o.isLoading) return <Loading rows={6} />;
  if (o.error) return <ErrorState error={o.error} />;
  const d = o.data;
  return (
    <div>
      <PageHeader eyebrow="Live view of the 9 parts" title="Architecture" icon={<MapIcon className="h-7 w-7" />} subtitle="The architecture diagram with what each part holds right now. Click a part to open the page that shows it." />
      <Card className="mb-6 h-[480px] overflow-hidden">
        <ReactFlow nodes={graph.nodes} edges={graph.edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.12 }} nodesConnectable={false} proOptions={{ hideAttribution: true }}>
          <Background gap={24} size={1} color="rgb(148 163 184 / 0.25)" />
        </ReactFlow>
      </Card>
      <div className="grid gap-4 xl:grid-cols-3">
        <Card><CardHeader title="Models" /><div className="p-5"><KeyVals items={[["LLM provider", d.models?.llm_provider], ["Embeddings", `${d.models?.embedding_model} (${d.models?.embedding_provider})`], ["Reranker", d.models?.reranking_model], ["Kaggle AI server", d.ai_server?.used ? (d.ai_server.reachable ? `${d.ai_server.model} · ${d.ai_server.requests ?? 0} requests` : "not reachable") : "not used"]]} cols={1} /></div></Card>
        <Card><CardHeader title="PostgreSQL rows" /><div className="flex flex-wrap gap-2 p-5">{Object.entries(d.tables || {}).map(([k, v]) => <Badge key={k} tone="indigo">{k}: {String(v)}</Badge>)}</div></Card>
        <Card><CardHeader title="Qdrant vectors" /><div className="flex flex-wrap gap-2 p-5">{Object.entries(d.vectors || {}).map(([k, v]) => <Badge key={k} tone="cyan">{k}: {String(v)}</Badge>)}</div></Card>
      </div>
    </div>
  );
}
