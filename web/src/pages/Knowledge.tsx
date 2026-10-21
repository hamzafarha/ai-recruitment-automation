import { useQuery } from "@tanstack/react-query";
import { Library } from "lucide-react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { ErrorState } from "../components/common";
import { DocumentBody } from "../components/domain/DocViewer";
import { Badge, Card, cx, Empty, Loading, PageHeader } from "../components/ui";
import { api } from "../lib/api";

export default function Knowledge() {
  const { docId } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const docs = useQuery({ queryKey: ["kb-docs"], queryFn: () => api.get("/knowledge/documents").then((r) => r.data as any[]) });
  const groups = (docs.data || []).reduce((acc: Record<string, any[]>, d) => { (acc[d.category_label] ||= []).push(d); return acc; }, {});
  const current = docId || docs.data?.[0]?.doc_id;
  return (
    <div>
      <PageHeader eyebrow="Diagram part 6 · RAG sources" title="HR knowledge base" icon={<Library className="h-7 w-7" />}
        subtitle="The 15 documents the RAG retrieves from: HR policies, recruitment rules (scoring rubric, experience counting, interviews), job profiles and skill definitions. Each is split by section into chunks, embedded with BGE-M3 and stored in Qdrant; the agents cite them by id (e.g. KB-RR-002)." />
      {docs.isLoading ? <Loading rows={6} /> : docs.error ? <ErrorState error={docs.error} /> : !docs.data?.length ? <Card><Empty title="No document found" /></Card> : (
        <div className="grid gap-5 xl:grid-cols-[300px_1fr]">
          <Card className="max-h-[80vh] overflow-y-auto p-3">
            {Object.entries(groups).map(([group, items]) => (
              <div key={group} className="mb-3">
                <div className="muted px-2 pb-1 text-[10px] font-semibold uppercase tracking-[0.16em]">{group}</div>
                {items.map((d) => (
                  <button key={d.doc_id} onClick={() => navigate(`/knowledge/${d.doc_id}`)} className={cx("w-full rounded-xl px-3 py-2 text-left transition", d.doc_id === current ? "bg-indigo-500/10 ring-1 ring-indigo-400/40" : "hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]")}>
                    <div className="flex items-center gap-2"><Badge tone="indigo">{d.doc_id}</Badge><Badge tone="slate">{d.language}</Badge></div>
                    <div className="mt-1 text-sm font-medium leading-snug">{d.title}</div>
                    <div className="muted text-[11px]">{d.chunks} chunks · {d.words} words</div>
                  </button>
                ))}
              </div>
            ))}
          </Card>
          <Card className="p-6">{current && <DocumentBody key={current} docId={current} section={params.get("section")} chunkId={params.get("chunk")} />}</Card>
        </div>
      )}
    </div>
  );
}
