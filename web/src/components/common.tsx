// Pieces several pages share: API-down state, job/candidate pickers, the CV submission form.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plug, Rocket, Workflow } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { api, ApiError, errorText, fileToBase64 } from "../lib/api";
import { Callout, Card, cx, Dropzone, Empty, Field, Spinner } from "./ui";

export function ErrorState({ error }: { error: unknown }) {
  const down = error instanceof ApiError && error.status === 0;
  return (
    <Card className="p-2">
      <Empty icon={<Plug className="h-6 w-6" />} title={down ? "The API is not reachable" : "Could not load this page"}>
        {down ? (
          <>Start it from the project folder: <code className="rounded bg-slate-500/10 px-1.5 py-0.5 font-mono text-xs">python -m uvicorn api.main:app --host 0.0.0.0 --port 8000</code> (and Docker: PostgreSQL, Qdrant). See the Run guide.</>
        ) : errorText(error)}
      </Empty>
    </Card>
  );
}

export const useJobs = () => useQuery({ queryKey: ["jobs"], queryFn: () => api.get("/jobs").then((r) => r.data as any[]) });
export const useCandidates = () => useQuery({ queryKey: ["candidates"], queryFn: () => api.get("/candidates").then((r) => r.data as any[]) });

export const jobLabel = (j: any) => `${j.title}${j.company ? ` — ${j.company}` : ""}`;

export function JobSelect({ value, onChange, openOnly, allowNone, noneLabel = "— choose a job —" }: { value: string; onChange: (v: string) => void; openOnly?: boolean; allowNone?: boolean; noneLabel?: string }) {
  const jobs = useJobs();
  const list = (jobs.data || []).filter((j) => !openOnly || j.status === "open");
  return (
    <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
      {(allowNone || !value) && <option value="">{noneLabel}</option>}
      {list.map((j) => <option key={j.id} value={j.id}>{jobLabel(j)}{j.status !== "open" ? " (closed)" : ""}</option>)}
    </select>
  );
}

export function CandidateSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const candidates = useCandidates();
  return (
    <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">— choose a candidate —</option>
      {(candidates.data || []).map((c) => <option key={c.id} value={c.id}>{c.full_name || "Unnamed"}{c.headline ? ` — ${c.headline}` : ""}</option>)}
    </select>
  );
}

/** A CV into the application pipeline: through n8n (one execution runs every step) or the API itself. */
export function SubmitCv({ source = "upload", onSubmitted, portal }: { source?: "upload" | "portal"; onSubmitted?: (applicationId: string, message?: string) => void; portal?: boolean }) {
  const client = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState("");
  const [route, setRoute] = useState<"api" | "n8n">("api");
  const [consent, setConsent] = useState(!portal);
  const submit = useMutation({
    mutationFn: async () => {
      const body = { filename: file!.name, content_base64: await fileToBase64(file!), source, consent, job_id: jobId || null };
      if (route === "n8n") {
        const r: any = await api.post("/applications/via-n8n", body);
        return { id: r.application_id, message: r.message };
      }
      const r: any = await api.post("/applications/submit", body);
      return { id: r.id, message: "Received: the API runs every step in the background (about a minute for the extraction on the Kaggle GPU, a few minutes for the screening agent)." };
    },
    onSuccess: (r) => { toast.success(r.message || "Application received"); client.invalidateQueries(); setFile(null); onSubmitted?.(r.id, r.message); },
    onError: (e) => toast.error(errorText(e)),
  });
  return (
    <div className="space-y-4">
      <Dropzone file={file} onFile={setFile} />
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Job applied to" hint="None = spontaneous application: the screening agent finds the open jobs that fit (grade ≥ 2, at most 2).">
          <JobSelect value={jobId} onChange={setJobId} openOnly allowNone noneLabel="No specific job (spontaneous application)" />
        </Field>
        <Field label="Run by" hint={route === "n8n" ? "The n8n workflow “Application pipeline” runs one node per step (n8n must be started and published)." : "The API runs the same steps itself, in the background."}>
          <div className="grid grid-cols-2 gap-2">
            {(["api", "n8n"] as const).map((r) => (
              <button key={r} type="button" onClick={() => setRoute(r)} className={cx("rounded-xl border px-3 py-2 text-sm font-medium transition", route === r ? "border-indigo-400 bg-indigo-500/10 text-indigo-700 dark:text-indigo-200" : "hairline")}>
                {r === "api" ? "The API" : "n8n workflow"}
              </button>
            ))}
          </div>
        </Field>
      </div>
      {portal && (
        <label className="flex items-start gap-3 text-sm">
          <input type="checkbox" className="mt-1 h-4 w-4 accent-indigo-500" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          <span>I accept that my CV is processed for this application only and kept at most 24 months, and that AI assists the recruiters — a human always decides (candidate data policy KB-HR-004).</span>
        </label>
      )}
      {route === "n8n" && <Callout tone="sky">n8n answers once the candidate is stored (after the extraction, about a minute); the screening continues in the same n8n execution.</Callout>}
      <div className="flex justify-end">
        <button className="btn-primary" disabled={!file || !consent || submit.isPending} onClick={() => submit.mutate()}>
          {submit.isPending ? <Spinner /> : portal ? <Rocket className="h-4 w-4" /> : <Workflow className="h-4 w-4" />} {portal ? "Send my application" : "Submit to the pipeline"}
        </button>
      </div>
    </div>
  );
}
