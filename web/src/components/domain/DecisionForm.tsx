// Human validation (diagram part 9): the recruiter's decision on an analysis. The decision resumes the
// application's workflow: e-mail to the candidate, and a video meeting when an interview is scheduled.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarPlus, CheckCircle2, PauseCircle, Send, Video, XCircle } from "lucide-react";
import { ReactNode, useState } from "react";
import { toast } from "sonner";
import { api, errorText } from "../../lib/api";
import { GRADE_LABELS } from "../../lib/format";
import { Callout, cx, Field, Spinner } from "../ui";

const OPTIONS: { id: string; label: string; desc: string; icon: ReactNode; tone: string }[] = [
  { id: "interview", label: "Schedule interview", desc: "E-mail + video meeting + calendar invite", icon: <CalendarPlus className="h-5 w-5" />, tone: "violet" },
  { id: "shortlist", label: "Shortlist", desc: "Next step, contacted within 5 days", icon: <CheckCircle2 className="h-5 w-5" />, tone: "emerald" },
  { id: "hold", label: "Put on hold", desc: "The workflow keeps waiting", icon: <PauseCircle className="h-5 w-5" />, tone: "slate" },
  { id: "reject", label: "Reject", desc: "A written reason is required", icon: <XCircle className="h-5 w-5" />, tone: "rose" },
];
const ACTIVE: Record<string, string> = {
  violet: "border-violet-400 bg-violet-500/10 text-violet-700 dark:text-violet-200",
  emerald: "border-emerald-400 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200",
  slate: "border-slate-400 bg-slate-500/10",
  rose: "border-rose-400 bg-rose-500/10 text-rose-700 dark:text-rose-200",
};

function inThreeDays(): string {
  const d = new Date(Date.now() + 3 * 86400000);
  d.setHours(10, 0, 0, 0);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T10:00`;
}

export function DecisionForm({ analysis, onDone }: { analysis: any; onDone?: (result: any) => void }) {
  const client = useQueryClient();
  const settings = useQuery({ queryKey: ["email-settings"], queryFn: () => api.get("/emails/settings") });
  const [decision, setDecision] = useState<string | null>(null);
  const [reviewer, setReviewer] = useState(() => { try { return localStorage.getItem("reviewer") || "recruiter"; } catch { return "recruiter"; } });
  const [grade, setGrade] = useState<string>("keep");
  const [at, setAt] = useState(inThreeDays());
  const [notes, setNotes] = useState("");
  const save = useMutation({
    mutationFn: () => api.post(`/analyses/${analysis.id}/review`, {
      decision, reviewer: reviewer || "recruiter", notes: notes || null,
      adjusted_grade: grade === "keep" ? null : Number(grade),
      interview_at: decision === "interview" ? at.replace("T", " ") : null,
    }),
    onSuccess: (result: any) => {
      try { localStorage.setItem("reviewer", reviewer); } catch { /* private window */ }
      const app = result.application;
      toast.success(app ? (app.status === "closed" ? "Decision saved: the candidate's e-mail is written and the application is closed" : app.status === "running" ? "Decision saved: n8n runs the last steps (e-mail to the candidate)" : "Decision saved: the application keeps waiting") : "Decision saved");
      client.invalidateQueries();
      onDone?.(result);
    },
    onError: (e) => toast.error(errorText(e)),
  });
  const s = settings.data;
  const report = analysis.report || {};
  return (
    <div className="space-y-4">
      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {OPTIONS.map((o) => (
          <button key={o.id} type="button" onClick={() => setDecision(o.id)}
            className={cx("rounded-2xl border p-3 text-left transition", decision === o.id ? ACTIVE[o.tone] : "hairline hover:border-indigo-400/50")}>
            <div className="flex items-center gap-2 text-sm font-semibold">{o.icon}{o.label}</div>
            <div className="muted mt-1 text-xs">{o.desc}</div>
          </button>
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Reviewer"><input className="input" value={reviewer} onChange={(e) => setReviewer(e.target.value)} /></Field>
        <Field label="Grade">
          <select className="input" value={grade} onChange={(e) => setGrade(e.target.value)}>
            <option value="keep">Keep {report.grade} (AI)</option>
            {[3, 2, 1, 0].map((g) => <option key={g} value={g}>Set to {g} · {GRADE_LABELS[g]}</option>)}
          </select>
        </Field>
        {decision === "interview" && <Field label="Interview date & time"><input type="datetime-local" className="input" value={at} onChange={(e) => setAt(e.target.value)} /></Field>}
      </div>
      <Field label={decision === "reject" ? "Reason (required, internal: never sent to the candidate)" : "Notes (internal)"}>
        <textarea className="input min-h-[80px]" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder={decision === "reject" ? "e.g. The must-have M2 (REST APIs) is not evidenced" : "Optional"} />
      </Field>
      {decision && decision !== "hold" && analysis.application && s && (
        <Callout tone={decision === "interview" ? "violet" : "sky"} icon={decision === "interview" ? <Video className="h-4 w-4" /> : <Send className="h-4 w-4" />}
          title={decision === "interview" ? "An interview invitation will be e-mailed with a video meeting" : "The answer will be e-mailed to the candidate"}>
          {s.mode === "smtp" ? <>Sent by SMTP from <b>{s.from}</b>{s.redirect_to ? <> to <b>{s.redirect_to}</b> (demo redirection)</> : " to the candidate's address"}.</> : <>EMAIL_MODE=outbox: written to the Outbox, not sent.</>}
          {decision === "interview" && <> Meeting: <b>{s.meeting_provider_used === "google" ? "Google Meet (Google Calendar sends the invitation)" : s.meeting_provider_used === "jitsi" ? "Jitsi Meet link + calendar invite (.ics)" : "no link"}</b>, {s.interview_minutes} min, {s.interview_timezone}.</>}
        </Callout>
      )}
      <div className="flex justify-end">
        <button className="btn-primary" disabled={!decision || save.isPending || (decision === "reject" && !notes.trim())} onClick={() => save.mutate()}>
          {save.isPending ? <Spinner /> : <CheckCircle2 className="h-4 w-4" />} Save the decision
        </button>
      </div>
    </div>
  );
}
