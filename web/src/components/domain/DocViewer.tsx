// The exact HR knowledge-base document a tool retrieved, opened on the cited section (drawer usable anywhere).
import { useQuery } from "@tanstack/react-query";
import { BookOpenText, Layers } from "lucide-react";
import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../lib/api";
import { Badge, Collapsible, cx, Drawer, ErrorBox, Loading, Markdown } from "../ui";

type Target = { docId: string; section?: string | null; chunkId?: string | null };
const DocContext = createContext<(t: Target) => void>(() => {});
export const useOpenDoc = () => useContext(DocContext);

function splitSections(markdown: string): { title: string | null; body: string }[] {
  const parts = ("\n" + markdown).split(/\n(?=## )/);
  return parts.filter((p) => p.trim()).map((p) => {
    const text = p.replace(/^\n/, "");
    if (text.startsWith("## ")) {
      const [heading] = text.split("\n", 1);
      return { title: heading.slice(3).trim(), body: text };
    }
    return { title: null, body: text };
  });
}

export function DocumentBody({ docId, section, chunkId }: Target) {
  const { data, isLoading, error } = useQuery({ queryKey: ["kb-doc", docId], queryFn: () => api.get(`/knowledge/documents/${encodeURIComponent(docId)}`) });
  const target = useRef<HTMLDivElement>(null);
  const sections = useMemo(() => splitSections(data?.content || ""), [data]);
  useEffect(() => {
    if (data && target.current) setTimeout(() => target.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
  }, [data, section]);
  if (isLoading) return <Loading label="Opening the document…" />;
  if (error) return <ErrorBox error={error} />;
  const norm = (s?: string | null) => (s || "").trim().toLowerCase();
  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <Badge tone="indigo">{data.doc_id}</Badge>
        <Badge tone="slate">{data.category_label}</Badge>
        <Badge tone="slate">{data.language.toUpperCase()}</Badge>
        {data.version && <Badge tone="slate">v{data.version}</Badge>}
        <Badge tone="cyan">{data.chunks} chunks indexed</Badge>
        <span className="muted text-xs">roles: {(data.roles || []).join(", ")} · file {data.file}</span>
      </div>
      {section && (
        <div className="mb-4 rounded-xl border border-amber-400/40 bg-amber-300/10 px-4 py-2 text-xs text-amber-700 dark:text-amber-200">
          The highlighted section <b>“{section}”</b>{chunkId ? <> (chunk <code>{chunkId}</code>)</> : null} is the passage the tool retrieved and gave to the model.
        </div>
      )}
      {sections.map((s, i) => {
        const hit = !!section && norm(s.title) === norm(section);
        return (
          <div key={i} ref={hit ? target : undefined} className={cx("scroll-mt-4", hit && "rounded-2xl bg-amber-300/10 px-4 py-1 ring-2 ring-amber-400/60")}>
            <Markdown>{s.body}</Markdown>
          </div>
        );
      })}
      <Collapsible className="mt-8" title={<><Layers className="h-4 w-4" /> How the RAG indexed this document ({data.chunks} chunks)</>}>
        <div className="space-y-3">
          {data.chunk_list.map((c: any) => (
            <div key={c.chunk_id} className={cx("rounded-xl border hairline p-3", c.chunk_id === chunkId && "ring-2 ring-amber-400/60")}>
              <div className="mb-1 flex items-center gap-2 text-xs"><Badge tone="indigo">{c.chunk_id}</Badge><span className="muted">{c.section}</span></div>
              <pre className="muted whitespace-pre-wrap font-mono text-[11px] leading-relaxed">{c.text}</pre>
            </div>
          ))}
        </div>
      </Collapsible>
    </div>
  );
}

export function DocProvider({ children }: { children: ReactNode }) {
  const [target, setTarget] = useState<Target | null>(null);
  const { data } = useQuery({ queryKey: ["kb-doc", target?.docId], queryFn: () => api.get(`/knowledge/documents/${encodeURIComponent(target!.docId)}`), enabled: !!target });
  return (
    <DocContext.Provider value={setTarget}>
      {children}
      <Drawer open={!!target} onClose={() => setTarget(null)} width="max-w-4xl"
        title={<span className="flex items-center gap-2"><BookOpenText className="h-5 w-5 text-indigo-500" /> {data?.title || target?.docId}</span>}
        subtitle="HR knowledge base · the exact document retrieved by the RAG (BGE-M3 in Qdrant, reordered by the BGE reranker)">
        {target && <DocumentBody {...target} />}
      </Drawer>
    </DocContext.Provider>
  );
}

export function DocLink({ docId, section, chunkId, children, className }: Target & { children?: ReactNode; className?: string }) {
  const open = useOpenDoc();
  if (!docId || docId === "?") return <span className={className}>{children ?? docId}</span>;
  return (
    <button onClick={(e) => { e.stopPropagation(); open({ docId, section, chunkId }); }}
      className={cx("inline-flex items-center gap-1 rounded-md font-medium text-indigo-600 underline decoration-indigo-400/40 underline-offset-2 hover:decoration-indigo-500 dark:text-indigo-300", className)}>
      <BookOpenText className="h-3.5 w-3.5" />{children ?? docId}
    </button>
  );
}
