import { BookOpen } from "lucide-react";
import { useMemo } from "react";
import guide from "../../../docs/USER_GUIDE.md?raw";
import { Card, Markdown, PageHeader } from "../components/ui";

const slug = (s: string) => s.toLowerCase().replace(/[^\w]+/g, "-").replace(/^-|-$/g, "");

export default function Guide() {
  const toc = useMemo(() => guide.split("\n").filter((l) => l.startsWith("## ")).map((l) => l.slice(3).trim()), []);
  return (
    <div>
      <PageHeader eyebrow="docs/USER_GUIDE.md" title="Run guide" icon={<BookOpen className="h-7 w-7" />} subtitle="How to install, start and use everything: the commands, the links, and where to go for each feature." />
      <div className="grid gap-6 xl:grid-cols-[240px_1fr]">
        <Card className="sticky top-20 hidden max-h-[80vh] self-start overflow-y-auto p-4 xl:block">
          <div className="muted mb-2 text-[10px] font-semibold uppercase tracking-[0.16em]">Contents</div>
          {toc.map((t) => <a key={t} href={`#${slug(t)}`} className="muted block rounded-lg px-2 py-1.5 text-xs hover:bg-slate-500/10 hover:text-[color:var(--text)]">{t}</a>)}
        </Card>
        <Card className="p-8">
          <GuideBody />
        </Card>
      </div>
    </div>
  );
}

function GuideBody() {
  // headings get ids so that the contents links work
  const parts = useMemo(() => guide.split(/\n(?=## )/), []);
  return (
    <>
      {parts.map((p, i) => {
        const title = p.startsWith("## ") ? p.split("\n", 1)[0].slice(3).trim() : null;
        return <div key={i} id={title ? slug(title) : undefined} className="scroll-mt-20"><Markdown>{p}</Markdown></div>;
      })}
    </>
  );
}
