import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Play, RefreshCw, Workflow } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";
import { ErrorState } from "../components/common";
import { PipelineStepper, StepPanel } from "../components/domain/Pipeline";
import { Badge, Callout, Card, GradeBadge, Loading, PageHeader, Progress, Spinner } from "../components/ui";
import { api, errorText } from "../lib/api";
import { APP_STATUS, DECISIONS, secs, when } from "../lib/format";

export default function ApplicationDetail() {
  const { id } = useParams();
  const client = useQueryClient();
  const app = useQuery({
    queryKey: ["application", id], queryFn: () => api.get(`/applications/${id}`),
    refetchInterval: (q) => ((q.state.data as any)?.status === "running" ? 3000 : 15000),
  });
  const outputs = useQuery({ queryKey: ["outputs", id, app.data?.updated_at], queryFn: () => api.get(`/applications/${id}/outputs`), enabled: !!app.data, placeholderData: (prev) => prev });
  const [selected, setSelected] = useState<string | null>(null);
  const cont = useMutation({
    mutationFn: () => api.post(`/applications/${id}/continue`),
    onSuccess: () => { toast.success("The workflow continued"); client.invalidateQueries(); },
    onError: (e) => toast.error(errorText(e)),
  });
  const a = app.data;
  // follow the workflow: show the step running, waiting or failed (until the user picks one)
  const focus = a?.steps?.find((s: any) => ["running", "failed", "waiting"].includes(s.status))?.step
    || [...(a?.steps || [])].reverse().find((s: any) => s.status === "done")?.step || "intake";
  useEffect(() => { if (a && selected === null) setSelected(a.status === "closed" ? "screen" : focus); }, [a, selected, focus]);
  if (app.isLoading) return <Loading label="Loading the application…" rows={6} />;
  if (app.error) return <ErrorState error={app.error} />;
  const step = a.steps.find((s: any) => s.step === (selected || focus)) || a.steps[0];
  const total = a.steps.reduce((t: number, s: any) => t + (s.status === "done" && s.step !== "validation" ? s.seconds || 0 : 0), 0);
  return (
    <div>
      <Link to="/pipeline" className="muted mb-4 inline-flex items-center gap-1.5 text-sm hover:text-[color:var(--text)]"><ArrowLeft className="h-4 w-4" /> All applications</Link>
      <PageHeader eyebrow={`Application ${String(a.id).slice(0, 8)} · ${a.source} · run by ${a.orchestrator}`} title={a.candidate_name || a.filename} icon={<Workflow className="h-7 w-7" />}
        subtitle={<>{a.applied_for} · received {when(a.created_at)}</>}
        actions={<>
          <Badge tone={APP_STATUS[a.status]?.tone} dot className="px-3 py-1 text-xs">{APP_STATUS[a.status]?.label}</Badge>
          {a.grade !== null && a.grade !== undefined && <GradeBadge grade={a.grade} />}
          {a.decision && <Badge tone={DECISIONS[a.decision]?.tone} className="px-3 py-1 text-xs">{DECISIONS[a.decision]?.label}</Badge>}
          <button className="btn-ghost btn-sm" onClick={() => client.invalidateQueries({ queryKey: ["application", id] })}><RefreshCw className="h-3.5 w-3.5" /></button>
        </>} />

      <Card className="mb-6 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 text-sm">
          <div className="flex items-center gap-3"><span className="font-semibold">{a.steps_done}/{a.steps_total} steps done</span><span className="muted">· {secs(total)} of processing</span>{a.status === "running" && <span className="muted flex items-center gap-1.5"><Spinner /> live</span>}</div>
          <div className="w-48"><Progress value={a.steps_done / a.steps_total} tone={a.status === "failed" ? "rose" : "indigo"} /></div>
        </div>
        <PipelineStepper steps={a.steps} selected={step.step} onSelect={setSelected} />
      </Card>

      {a.status === "failed" && (
        <Callout tone="rose" title={`Stopped at «${a.stage_label}»`} className="mb-6">
          <div>{a.error}</div>
          <button className="btn-primary btn-sm mt-3" disabled={cont.isPending} onClick={() => cont.mutate()}>{cont.isPending ? <Spinner /> : <Play className="h-3.5 w-3.5" />} Continue the workflow from this step (with the API)</button>
        </Callout>
      )}
      {a.status === "waiting_review" && step.step !== "validation" && (
        <Callout tone="amber" title="Waiting for a recruiter's decision" className="mb-6">
          <button className="font-semibold underline" onClick={() => setSelected("validation")}>Decide now</button> — the decision resumes this workflow: the candidate is e-mailed, with a video meeting for an interview.
        </Callout>
      )}

      <Card className="p-6"><StepPanel app={a} step={step} outputs={outputs.data} /></Card>
    </div>
  );
}
