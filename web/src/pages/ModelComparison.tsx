import { useMutation, useQuery } from "@tanstack/react-query";
import { FileSearch, GitCompareArrows, Sparkles, UserSearch } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { CandidateSelect } from "../components/common";
import { ExtractionEval, FinetuneEval } from "../components/domain/EvalCharts";
import { ProfileCompare } from "../components/domain/Profile";
import { Badge, Callout, Card, CardHeader, CodeBlock, Collapsible, Dropzone, Empty, KeyVals, Loading, PageHeader, Spinner, Tabs } from "../components/ui";
import { api, errorText } from "../lib/api";
import { secs } from "../lib/format";

function LiveCompare() {
  const [file, setFile] = useState<File | null>(null);
  const run = useMutation({
    mutationFn: () => { const f = new FormData(); f.append("file", file!); f.append("compare_base", "true"); return api.upload("/observe/extraction", f); },
    onError: (e) => toast.error(errorText(e)),
  });
  const r = run.data;
  return (
    <div className="space-y-5">
      <Card className="p-5">
        <div className="muted mb-4 text-sm">Upload any CV: it is read (PDF text layer, Word or OCR), then the <b>same prompt</b> is sent twice to Qwen2.5-7B — once with our QLoRA adapter, once without. Nothing is stored. Two generations: about 2-3 minutes on the Kaggle GPU.</div>
        <Dropzone file={file} onFile={setFile} />
        <div className="mt-4 flex justify-end"><button className="btn-primary" disabled={!file || run.isPending} onClick={() => run.mutate()}>{run.isPending ? <Spinner /> : <GitCompareArrows className="h-4 w-4" />} Compare base vs fine-tuned</button></div>
        {run.isPending && <div className="mt-4"><Loading label="Reading the CV, then running both models…" /></div>}
      </Card>
      {r && (
        <>
          <Card className="p-5">
            <div className="mb-3 text-sm font-semibold">Document processing</div>
            <KeyVals cols={4} items={[["Reader", r.parsing.engine], ["Input", r.parsing.input_type], ["Pages", r.parsing.pages.length], ["Characters", r.parsing.characters], ["Reading", secs(r.parsing.seconds)], ["Adapter on the server", r.adapter ? <Badge key="a" tone="indigo">{String(r.adapter).split("/").pop()}</Badge> : "none"], ["Prompt", `${r.prompt.user_chars} characters${r.prompt.cv_truncated ? " (CV cut)" : ""}`]]} />
          </Card>
          <Card className="p-5"><ProfileCompare result={r} /></Card>
          <Collapsible title="The exact prompt and both raw answers">
            <div className="space-y-3">
              <CodeBlock title="system prompt" code={r.prompt.system} maxHeight="max-h-48" />
              <CodeBlock title="user prompt (same for both)" code={r.prompt.user} maxHeight="max-h-72" />
              <div className="grid gap-3 xl:grid-cols-2">
                <CodeBlock title="base model — raw answer" code={r.base?.raw_answer || ""} />
                <CodeBlock title="+ QLoRA adapter — raw answer" code={r.finetuned?.raw_answer || ""} />
              </div>
            </div>
          </Collapsible>
        </>
      )}
    </div>
  );
}

function StoredCompare() {
  const [id, setId] = useState("");
  const cached = useQuery({ queryKey: ["compare-cached", id], queryFn: () => api.get(`/candidates/${id}/compare-models`), enabled: !!id });
  const run = useMutation({ mutationFn: () => api.post(`/candidates/${id}/compare-models`), onSuccess: () => cached.refetch(), onError: (e) => toast.error(errorText(e)) });
  const result = run.data || cached.data?.comparison;
  return (
    <div className="space-y-5">
      <Card className="p-5">
        <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
          <div><div className="label">Candidate (stored with the fine-tuned profile)</div><CandidateSelect value={id} onChange={setId} /></div>
          <button className="btn-primary" disabled={!id || run.isPending} onClick={() => run.mutate()}>{run.isPending ? <Spinner /> : <Sparkles className="h-4 w-4" />} Run the base model on this CV</button>
        </div>
        <div className="muted mt-2 text-xs">One generation of the base model on the stored CV text; the fine-tuned side is the profile written when the candidate was stored. The result is kept with the candidate and shown in the application's extraction step.</div>
      </Card>
      {result ? <Card className="p-5"><ProfileCompare result={result} /></Card> : id ? <Card><Empty icon={<UserSearch className="h-6 w-6" />} title="Not compared yet" /></Card> : null}
    </div>
  );
}

export default function ModelComparison() {
  const [tab, setTab] = useState<"offline" | "live" | "stored" | "training">("offline");
  const ex = useQuery({ queryKey: ["eval-extraction"], queryFn: () => api.get("/evaluation/extraction") });
  const ft = useQuery({ queryKey: ["eval-finetune"], queryFn: () => api.get("/evaluation/finetune"), enabled: tab === "training" });
  return (
    <div>
      <PageHeader eyebrow="DSO1 · fine-tuning" title="Qwen2.5 base vs Qwen2.5 + QLoRA" icon={<GitCompareArrows className="h-7 w-7" />}
        subtitle="What our QLoRA fine-tuning (1000 CVs, Kaggle 2× T4) changes in CV extraction: measured on the frozen evaluation set, live on any CV, or on a stored candidate. Same prompt, same code, same scoring — only the adapter differs." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "offline", label: "Evaluation (frozen set)", icon: <FileSearch className="h-4 w-4" /> }, { id: "live", label: "Live on a CV", icon: <GitCompareArrows className="h-4 w-4" /> }, { id: "stored", label: "On a stored candidate", icon: <UserSearch className="h-4 w-4" /> }, { id: "training", label: "The training run", icon: <Sparkles className="h-4 w-4" /> }]} />
      {tab === "offline" && (ex.isLoading ? <Loading rows={6} /> : <><Callout tone="indigo" className="mb-5">Runs: <b>{ex.data?.before}</b> (base) vs <b>{ex.data?.after}</b> (fine-tuned), {ex.data?.runs?.find((x: any) => x.name === ex.data?.after)?.documents} CVs of the dev split — the test split stays untouched for the final evaluation.</Callout><ExtractionEval data={ex.data} /></>)}
      {tab === "live" && <LiveCompare />}
      {tab === "stored" && <StoredCompare />}
      {tab === "training" && (ft.isLoading ? <Loading /> : <FinetuneEval data={ft.data} />)}
    </div>
  );
}

