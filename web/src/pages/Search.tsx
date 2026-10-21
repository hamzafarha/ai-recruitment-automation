import { useMutation } from "@tanstack/react-query";
import { BookOpen, Briefcase, Search as SearchIcon, Users } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { DocLink } from "../components/domain/DocViewer";
import { Badge, Card, Empty, PageHeader, Progress, Spinner, Tabs } from "../components/ui";
import { api, errorText } from "../lib/api";

type Kind = "knowledge" | "candidates" | "jobs";
const PLACEHOLDER: Record<Kind, string> = { knowledge: "How are internships counted? · Politique de télétravail", candidates: "built RAG pipelines with LangChain", jobs: "Kubernetes and Terraform" };

export default function Search() {
  const [kind, setKind] = useState<Kind>("knowledge");
  const [q, setQ] = useState("");
  const search = useMutation({ mutationFn: () => api.get(`/search/${kind}?q=${encodeURIComponent(q)}&top_k=${kind === "knowledge" ? 6 : 8}`), onError: (e) => toast.error(errorText(e)) });
  const rows: any[] = search.data?.data || [];
  return (
    <div>
      <PageHeader eyebrow="Diagram parts 5-6" title="Semantic search" icon={<SearchIcon className="h-7 w-7" />} subtitle="BGE-M3 embeddings in Qdrant, reordered by the BGE cross-encoder reranker — the same search the agents' tools use. Works across English and French." />
      <Tabs value={kind} onChange={(k) => { setKind(k); search.reset(); }} tabs={[{ id: "knowledge", label: "HR knowledge base", icon: <BookOpen className="h-4 w-4" /> }, { id: "candidates", label: "Candidates", icon: <Users className="h-4 w-4" /> }, { id: "jobs", label: "Jobs", icon: <Briefcase className="h-4 w-4" /> }]} />
      <Card className="mb-5 p-4">
        <form onSubmit={(e) => { e.preventDefault(); q.trim().length >= 2 && search.mutate(); }} className="flex gap-2">
          <div className="relative flex-1"><SearchIcon className="muted absolute left-3 top-2.5 h-4 w-4" /><input className="input pl-9" placeholder={PLACEHOLDER[kind]} value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <button className="btn-primary" disabled={q.trim().length < 2 || search.isPending}>{search.isPending ? <Spinner /> : <SearchIcon className="h-4 w-4" />} Search</button>
        </form>
      </Card>
      {search.data && !rows.length && <Card><Empty title="Nothing found">Index the knowledge base and load data first (Dashboard → Set up).</Empty></Card>}
      <div className="space-y-3">
        {rows.map((r, i) => (
          <Card key={i} className="p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="muted font-mono text-xs">#{i + 1}</span>
              {kind === "knowledge" ? <><DocLink docId={r.doc_id} section={r.section}>{r.doc_id}</DocLink><span className="font-medium">{r.title}</span><span className="muted text-sm">· {r.section}</span><Badge tone="slate">{r.language}</Badge></>
                : <><span className="font-semibold">{r.name}</span>{r.label && <span className="muted text-sm">· {r.label}</span>}<Badge tone="cyan">{r.section}</Badge></>}
              <div className="ml-auto flex w-40 items-center gap-2"><Progress value={Math.max(0, Math.min(1, r.score))} /><span className="font-mono text-xs">{Number(r.score).toFixed(3)}</span></div>
            </div>
            <p className="muted mt-2 whitespace-pre-wrap text-sm leading-relaxed">{r.text || r.evidence}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}
