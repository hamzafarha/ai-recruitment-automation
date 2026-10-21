// An e-mail to the candidate (acknowledgement, answer, interview invitation with its video meeting).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CalendarClock, ExternalLink, Mail, Paperclip, Send, Video } from "lucide-react";
import { toast } from "sonner";
import { api, errorText } from "../../lib/api";
import { EMAIL_STATUS, when } from "../../lib/format";
import { Badge, Callout, cx } from "../ui";

export function MeetingCard({ meeting }: { meeting: any }) {
  if (!meeting) return null;
  const start = meeting.start ? new Date(meeting.start) : null;
  const google = meeting.provider === "google";
  return (
    <div className="relative overflow-hidden rounded-2xl border border-violet-400/30 bg-gradient-to-br from-violet-500/10 via-indigo-500/5 to-cyan-500/10 p-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className={cx("grid h-12 w-12 place-items-center rounded-2xl text-white shadow-lg", google ? "bg-gradient-to-br from-emerald-500 to-sky-500" : "bg-gradient-to-br from-indigo-500 to-violet-500")}>
          <Video className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-sm font-semibold">
            {meeting.title || "Interview"}
            <Badge tone={google ? "emerald" : "violet"}>{google ? "Google Meet" : meeting.provider === "jitsi" ? "Jitsi Meet" : meeting.provider}</Badge>
            {meeting.invitation_by === "google" && <Badge tone="emerald">Google sent the invitation</Badge>}
          </div>
          <div className="muted mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            <span className="flex items-center gap-1"><CalendarClock className="h-3.5 w-3.5" />{start ? start.toLocaleString(undefined, { weekday: "long", day: "numeric", month: "long", hour: "2-digit", minute: "2-digit", timeZone: meeting.timezone || undefined }) : "—"} ({meeting.timezone}) · {meeting.minutes} min</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {meeting.url && <a href={meeting.url} target="_blank" rel="noreferrer" className="btn-primary btn-sm"><Video className="h-3.5 w-3.5" /> Join</a>}
          {meeting.calendar_link && <a href={meeting.calendar_link} target="_blank" rel="noreferrer" className="btn-ghost btn-sm"><ExternalLink className="h-3.5 w-3.5" /> Calendar event</a>}
        </div>
      </div>
      {meeting.url && <div className="mt-3 truncate rounded-lg bg-black/5 px-3 py-1.5 font-mono text-xs dark:bg-white/5">{meeting.url}</div>}
      {(meeting.notes || []).map((n: string, i: number) => <div key={i} className="mt-2 text-xs text-amber-700 dark:text-amber-300">{n}</div>)}
    </div>
  );
}

export function EmailCard({ email, compact }: { email: any; compact?: boolean }) {
  const client = useQueryClient();
  const resend = useMutation({
    mutationFn: () => api.post(`/emails/${email.id}/send`),
    onSuccess: (e: any) => { toast.success(`E-mail ${EMAIL_STATUS[e.status]?.label.toLowerCase() || e.status}`); client.invalidateQueries(); },
    onError: (e) => toast.error(errorText(e)),
  });
  const status = EMAIL_STATUS[email.status] || { label: email.status, tone: "slate" };
  return (
    <div className="overflow-hidden rounded-2xl border hairline">
      <div className="flex flex-wrap items-center gap-3 border-b hairline surface-2 px-4 py-3">
        <Mail className="h-4 w-4 text-indigo-500" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{email.subject}</div>
          <div className="muted mt-0.5 text-xs">
            To <b className="text-[color:var(--text)]">{email.to || "(no address)"}</b>
            {email.redirected && <> · redirected from <span className="line-through">{email.candidate_address}</span> (EMAIL_REDIRECT_TO)</>}
            {" · "}{email.kind === "acknowledgement" ? "acknowledgement" : `answer · ${email.decision}`} · {when(email.sent_at || email.created_at)}
          </div>
        </div>
        <Badge tone={status.tone} dot>{status.label}</Badge>
        <Badge tone="slate">{email.mode}</Badge>
        {(email.status === "outbox" || email.status === "failed") && (
          <button onClick={() => resend.mutate()} disabled={resend.isPending} className="btn-ghost btn-sm"><Send className="h-3.5 w-3.5" /> Send now</button>
        )}
      </div>
      {email.error && <div className="border-b hairline px-4 py-2 text-xs text-rose-600 dark:text-rose-300">{email.error}</div>}
      {!compact && email.text && <pre className="whitespace-pre-wrap px-4 py-3 font-sans text-[13px] leading-relaxed">{email.text}</pre>}
      {email.meeting && <div className="px-4 pb-4"><MeetingCard meeting={email.meeting} /></div>}
      {!!(email.attachments || []).length && (
        <div className="muted flex items-center gap-2 border-t hairline px-4 py-2 text-xs"><Paperclip className="h-3.5 w-3.5" />{email.attachments.map((a: any) => a.filename || a).join(", ")} (calendar invitation)</div>
      )}
    </div>
  );
}

export function OutboxHint({ status }: { status?: string }) {
  if (status !== "outbox") return null;
  return <Callout tone="sky">EMAIL_MODE=outbox: the e-mail is kept in the app (Outbox page), not sent. Set the SMTP settings in <code>.env</code> and <code>EMAIL_MODE=smtp</code> to really send it (Run guide § E-mails).</Callout>;
}
