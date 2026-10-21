import { useQuery } from "@tanstack/react-query";
import { Mail, Settings2, Video } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { ErrorState } from "../components/common";
import { EmailCard } from "../components/domain/Email";
import { Badge, Callout, Card, CardHeader, cx, Empty, KeyVals, Loading, PageHeader, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { EMAIL_STATUS, when } from "../lib/format";

export default function Outbox() {
  const [status, setStatus] = useState("all");
  const [open, setOpen] = useState<string | null>(null);
  const settings = useQuery({ queryKey: ["email-settings"], queryFn: () => api.get("/emails/settings") });
  const list = useQuery({ queryKey: ["emails", status], queryFn: () => api.get(`/emails${status === "all" ? "" : `?status=${status}`}`), refetchInterval: 15_000 });
  const s = settings.data;
  const rows: any[] = list.data?.data || [];
  const counts = list.data?.by_status || {};
  return (
    <div>
      <PageHeader eyebrow="Diagram part 7 · notifications" title="E-mails & meetings" icon={<Mail className="h-7 w-7" />}
        subtitle="Every e-mail the pipeline wrote to a candidate: the acknowledgement, then the answer after the recruiter's decision — an interview comes with a video meeting (Google Meet through Google Calendar, or Jitsi) and a calendar invitation." />
      {s && (
        <Card className="mb-6">
          <CardHeader title="How e-mails are sent" subtitle="Settings of .env (restart the API after a change) — Run guide § E-mails and Google Meet" icon={<Settings2 className="h-4 w-4" />} />
          <div className="space-y-4 p-5">
            <KeyVals cols={4} items={[
              ["Mode", s.mode === "smtp" ? <Badge key="m" tone="emerald" dot>SMTP — really sent</Badge> : <Badge key="m" tone="sky" dot>Outbox — kept here, not sent</Badge>],
              ["From", s.from], ["SMTP", s.mode === "smtp" ? `${s.smtp_host}:${s.smtp_port} · ${s.smtp_user || "no login"}${s.smtp_password_set ? " · password set" : " · no password"}` : "—"],
              ["Redirection (demo)", s.redirect_to ? <Badge key="r" tone="amber">every e-mail → {s.redirect_to}</Badge> : "off: to the candidate's address"],
              ["Acknowledgement e-mail", s.acknowledge ? "on" : "off"],
              ["Interview meetings", <span key="p" className="flex items-center gap-1.5"><Video className="h-3.5 w-3.5" />{s.meeting_provider_used === "google" ? "Google Meet (Calendar)" : s.meeting_provider_used === "jitsi" ? "Jitsi Meet" : "none"} <span className="muted">(MEETING_PROVIDER={s.meeting_provider})</span></span>],
              ["Google Calendar", s.google_ready ? <Badge key="g" tone="emerald" dot>authorised</Badge> : <span key="g" className="muted">not set up — python scripts/google_calendar_setup.py</span>],
              ["Interview", `${s.interview_minutes} min · ${s.interview_timezone}`],
            ]} />
            {s.mode !== "smtp" && <Callout tone="sky">Outbox mode: nothing leaves this computer. To send for real, set <code>EMAIL_MODE=smtp</code> with a Gmail App password (and <code>EMAIL_REDIRECT_TO</code> = your own address for demos: the evaluation CVs belong to real people), then use “Send now” on any e-mail below.</Callout>}
          </div>
        </Card>
      )}
      <Tabs value={status} onChange={setStatus} tabs={[{ id: "all", label: "All", count: Object.values(counts).reduce((a: number, b: any) => a + Number(b), 0) as number }, ...Object.entries(EMAIL_STATUS).map(([k, v]) => ({ id: k, label: v.label, count: counts[k] || 0 }))]} />
      {list.isLoading ? <Loading rows={5} /> : list.error ? <ErrorState error={list.error} /> : !rows.length ? (
        <Card><Empty icon={<Mail className="h-6 w-6" />} title="No e-mail yet">The pipeline writes the acknowledgement when a CV is stored, and the answer after a recruiter's decision (<Link className="underline" to="/validation">Human validation</Link>).</Empty></Card>
      ) : (
        <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
          <Card className="max-h-[78vh] overflow-y-auto p-2">
            {rows.map((e) => (
              <button key={e.id} onClick={() => setOpen(e.id)} className={cx("w-full rounded-xl px-3 py-2.5 text-left transition", e.id === (open || rows[0]?.id) ? "bg-indigo-500/10 ring-1 ring-indigo-400/40" : "hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]")}>
                <div className="flex items-center justify-between gap-2"><span className="truncate text-sm font-semibold">{e.subject}</span><Badge tone={EMAIL_STATUS[e.status]?.tone}>{EMAIL_STATUS[e.status]?.label}</Badge></div>
                <div className="muted mt-0.5 truncate text-xs">to {e.to || "—"} · {when(e.created_at)}</div>
                <div className="mt-1 flex gap-1">{e.decision && <Badge tone="violet">{e.decision}</Badge>}{e.meeting && <Badge tone="emerald"><Video className="h-3 w-3" />{e.meeting.provider}</Badge>}</div>
              </button>
            ))}
          </Card>
          <div>{(open || rows[0]?.id) && <OutboxDetail key={open || rows[0].id} id={open || rows[0].id} />}</div>
        </div>
      )}
    </div>
  );
}

function OutboxDetail({ id }: { id: string }) {
  const email = useQuery({ queryKey: ["email", id], queryFn: () => api.get(`/emails/${id}`) });
  if (email.isLoading || !email.data) return <Card className="p-6"><Loading /></Card>;
  return (
    <div className="space-y-3">
      <EmailCard email={email.data} />
      {email.data.application_id && <Link className="btn-ghost btn-sm" to={`/pipeline/${email.data.application_id}`}>Open its application</Link>}
      {email.data.ics && <Card className="p-4"><div className="mb-2 text-sm font-semibold">Calendar invitation (invite.ics)</div><pre className="muted max-h-60 overflow-auto font-mono text-[11px]">{email.data.ics}</pre></Card>}
    </div>
  );
}
