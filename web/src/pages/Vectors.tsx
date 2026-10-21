import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Network, Search } from "lucide-react";
import { useState } from "react";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { toast } from "sonner";
import { ErrorState } from "../components/common";
import { DocLink } from "../components/domain/DocViewer";
import { Badge, Card, CardHeader, CodeBlock, Drawer, Empty, KeyVals, Loading, PageHeader, Spinner, Stat, Tabs } from "../components/ui";
import { api, errorText } from "../lib/api";
import { axisProps, PALETTE, tooltipStyle } from "../lib/chart";
import { secs } from "../lib/format";

const COLLECTIONS = ["hr_knowledge", "candidates", "jobs"];

function MapView({ name }: { name: string }) {
  const map = useQuery({ queryKey: ["vector-map", name], queryFn: () => api.get(`/observe/collections/${name}/map?limit=400`) });
  if (map.isLoading) return <Loading label="Projecting the vectors (PCA)…" />;
  if (map.error) return <ErrorState error={map.error} />;
  const groups: Record<string, any[]> = {};
  (map.data?.points || []).forEach((p: any) => (groups[p.group] ||= []).push(p));
  const names = Object.keys(groups);
  return !names.length ? <Empty title="Not enough points to draw a map" /> : (
    <div>
      <div className="muted mb-2 text-xs">PCA of the BGE-M3 vectors (1024 dimensions → 2): close points = similar texts. Explained variance: {(map.data.explained_variance || []).map((v: number) => `${(v * 100).toFixed(1)} %`).join(" + ")} · colored by {map.data.color_by}</div>
      <div className="h-[460px]">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
            <CartesianGrid strokeOpacity={0.1} />
            <XAxis type="number" dataKey="x" {...axisProps} name="PC1" />
            <YAxis type="number" dataKey="y" {...axisProps} name="PC2" />
            <ZAxis range={[60, 60]} />
            <Tooltip {...tooltipStyle} content={({ payload }: any) => payload?.[0] ? <div style={tooltipStyle.contentStyle as any} className="max-w-xs p-2"><div className="font-semibold">{payload[0].payload.label}</div><div className="mt-1 opacity-80">{payload[0].payload.text}</div></div> : null} />
            {names.length <= 16 && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {names.map((g, i) => <Scatter key={g} name={g} data={groups[g]} fill={PALETTE[i % PALETTE.length]} fillOpacity={0.8} />)}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Browse({ name }: { name: string }) {
  const [offset, setOffset] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const points = useQuery({ queryKey: ["points", name, offset], queryFn: () => api.get(`/observe/collections/${name}/points?limit=20${offset ? `&offset=${offset}` : ""}`) });
  const point = useQuery({ queryKey: ["point", name, open], queryFn: () => api.get(`/observe/collections/${name}/points/${open}`), enabled: !!open });
  return (
    <div>
      {points.isLoading ? <Loading /> : (
        <div className="space-y-2">
          {(points.data?.points || []).map((p: any) => (
            <button key={p.id} onClick={() => setOpen(p.id)} className="w-full rounded-xl border hairline p-3 text-left text-xs transition hover:border-indigo-400/50">
              <div className="mb-1 flex items-center gap-2"><Badge tone="cyan">{p.payload.section || p.payload.category}</Badge><span className="font-medium">{p.label}</span>{p.payload.chunk_id && <span className="muted font-mono">{p.payload.chunk_id}</span>}</div>
              <div className="muted line-clamp-2">{p.payload.text}</div>
            </button>
          ))}
          <div className="flex gap-2 pt-2"><button className="btn-ghost btn-sm" onClick={() => setOffset(null)} disabled={!offset}>First page</button><button className="btn-ghost btn-sm" disabled={!points.data?.next_offset} onClick={() => setOffset(points.data.next_offset)}>Next page</button></div>
        </div>
      )}
      <Drawer open={!!open} onClose={() => setOpen(null)} title={point.data?.label || "Point"} subtitle={`${name} · ${open}`}>
        {point.isLoading || !point.data ? <Loading /> : (
          <div className="space-y-4">
            <KeyVals cols={3} items={[["Dimension", point.data.dimension], ["Norm", point.data.norm], ["Collection", name]]} />
            {point.data.payload.document_id && <DocLink docId={point.data.payload.document_id} section={point.data.payload.section} chunkId={point.data.payload.chunk_id}>Open {point.data.payload.document_id}</DocLink>}
            <CodeBlock title="payload" code={JSON.stringify(point.data.payload, null, 2)} />
            <div><div className="label">The vector ({point.data.dimension} values)</div>
              <div className="grid grid-cols-[repeat(32,minmax(0,1fr))] gap-px overflow-hidden rounded-lg">{point.data.vector.slice(0, 1024).map((v: number, i: number) => <div key={i} title={`${i}: ${v}`} className="aspect-square" style={{ background: v >= 0 ? `rgba(99,102,241,${Math.min(1, Math.abs(v) * 12)})` : `rgba(244,63,94,${Math.min(1, Math.abs(v) * 12)})` }} />)}</div>
              <div className="muted mt-1 text-[10px]">Blue = positive, red = negative, darker = larger.</div>
            </div>
          </div>
        )}
      </Drawer>
    </div>
  );
}

function Retrieval() {
  const [q, setQ] = useState("How are internships counted in years of experience?");
  const [collection, setCollection] = useState("hr_knowledge");
  const run = useMutation({ mutationFn: () => api.get(`/observe/retrieval?q=${encodeURIComponent(q)}&collection=${collection}&top_k=5`), onError: (e) => toast.error(errorText(e)) });
  const r = run.data;
  return (
    <div className="space-y-4">
      <form onSubmit={(e) => { e.preventDefault(); run.mutate(); }} className="grid gap-2 md:grid-cols-[1fr_180px_auto]">
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} />
        <select className="input" value={collection} onChange={(e) => setCollection(e.target.value)}>{COLLECTIONS.map((c) => <option key={c}>{c}</option>)}</select>
        <button className="btn-primary" disabled={run.isPending || q.length < 2}>{run.isPending ? <Spinner /> : <Search className="h-4 w-4" />} Search step by step</button>
      </form>
      {r && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Stat label="1 · Query embedding" value={`${r.embedding.dimension} dims`} hint={`${r.embedding.model} · ${secs(r.seconds.embedding)}`} tone="indigo" />
            <Stat label="2 · Qdrant (cosine)" value={`${r.dense.length} hits`} hint={secs(r.seconds.qdrant)} tone="cyan" />
            <Stat label="3 · BGE reranker" value={r.reranker ? "reordered" : "off"} hint={`${r.reranker || "—"} · ${secs(r.seconds.reranking)}`} tone="violet" />
          </div>
          <div className="muted font-mono text-[11px]">first values: [{r.embedding.first_values.join(", ")}, …] · norm {r.embedding.norm}</div>
          <div className="grid gap-4 xl:grid-cols-2">
            {[["Dense order (cosine)", r.dense], ["After the reranker (what the agents use)", r.reranked]].map(([title, rows]: any) => (
              <Card key={title}>
                <CardHeader title={title} />
                <div className="space-y-2 p-4">{rows.map((h: any) => (
                  <div key={h.id} className="rounded-xl border hairline p-3 text-xs">
                    <div className="mb-1 flex flex-wrap items-center gap-2"><span className="font-mono">#{h.rerank_rank ?? h.dense_rank}</span><span className="font-medium">{h.label}</span><Badge tone="cyan">cos {h.cosine}</Badge>{h.rerank_score !== undefined && h.rerank_score !== null && <Badge tone="violet">rerank {h.rerank_score}</Badge>}
                      {h.moved > 0 && <span className="flex items-center text-emerald-500"><ArrowUp className="h-3 w-3" />{h.moved}</span>}{h.moved < 0 && <span className="flex items-center text-rose-500"><ArrowDown className="h-3 w-3" />{-h.moved}</span>}</div>
                    <div className="muted line-clamp-3">{h.text}</div>
                  </div>
                ))}</div>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function Vectors() {
  const [tab, setTab] = useState<"collections" | "browse" | "map" | "search">("collections");
  const [name, setName] = useState("hr_knowledge");
  const cols = useQuery({ queryKey: ["collections"], queryFn: () => api.get("/observe/collections") });
  return (
    <div>
      <PageHeader eyebrow="Diagram parts 5-6" title="Vectors & retrieval (RAG)" icon={<Network className="h-7 w-7" />} subtitle="The three Qdrant collections (BGE-M3, 1024 dimensions, cosine), their points, a 2D map of the vectors, and one search step by step: embedding → Qdrant → reranker." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "collections", label: "Collections" }, { id: "browse", label: "Browse points" }, { id: "map", label: "2D map" }, { id: "search", label: "One search, step by step" }]} />
      {tab !== "search" && tab !== "collections" && <div className="mb-4 flex gap-2">{COLLECTIONS.map((c) => <button key={c} onClick={() => setName(c)} className={`chip ${c === name ? "border-indigo-400 text-indigo-600 dark:text-indigo-300" : ""}`}>{c}</button>)}</div>}
      {tab === "collections" && (cols.isLoading ? <Loading /> : cols.error ? <ErrorState error={cols.error} /> : (
        <div className="grid gap-4 md:grid-cols-3">{(cols.data?.data || []).map((c: any) => <Stat key={c.name} label={c.name} value={c.points} hint={`${c.dimension ?? "—"} dims · ${c.distance || "—"} · ${c.status}`} tone="cyan" />)}</div>
      ))}
      {tab === "browse" && <Card className="p-5"><Browse key={name} name={name} /></Card>}
      {tab === "map" && <Card className="p-5"><MapView key={name} name={name} /></Card>}
      {tab === "search" && <Card className="p-5"><Retrieval /></Card>}
    </div>
  );
}
