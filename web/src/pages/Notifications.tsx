import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bell, CheckCheck, ExternalLink, Workflow } from "lucide-react";
import { Link } from "react-router-dom";
import { ErrorState } from "../components/common";
import { Badge, Callout, Card, CardHeader, cx, Dot, Empty, Loading, PageHeader } from "../components/ui";
import { api } from "../lib/api";
import { ago, secs } from "../lib/format";

export default function Notifications() {
  const client = useQueryClient();
  const list = useQuery({ queryKey: ["notifications"], queryFn: () => api.get("/notifications?limit=200"), refetchInterval: 15_000 });
  const n8n = useQuery({ queryKey: ["n8n"], queryFn: () => api.get("/observe/n8n"), refetchInterval: 30_000 });
  const readAll = useMutation({ mutationFn: () => api.post("/notifications/read-all"), onSuccess: () => client.invalidateQueries() });
  const read = useMutation({ mutationFn: (id: string) => api.post(`/notifications/${id}/read`), onSuccess: () => client.invalidateQueries() });
  const o = n8n.data || {};
  return (
    <div>
      <PageHeader eyebrow="Diagram parts 2, 7" title="Notifications & n8n" icon={<Bell className="h-7 w-7" />} subtitle="Every event of the app (application received, review needed, decision, answer e-mailed…) and the n8n workflows that orchestrate the pipeline."
        actions={<><button className="btn-ghost" onClick={() => readAll.mutate()}><CheckCheck className="h-4 w-4" /> Mark all read</button><a className="btn-ghost" href={o.url || "http://localhost:5678"} target="_blank" rel="noreferrer"><ExternalLink className="h-4 w-4" /> Open n8n</a></>} />
      <div className="grid gap-5 xl:grid-cols-[1fr_400px]">
        <Card className="p-3">
          {list.isLoading ? <Loading rows={6} /> : list.error ? <ErrorState error={list.error} /> : !list.data?.data?.length ? <Empty icon={<Bell className="h-6 w-6" />} title="No notification yet" /> : list.data.data.map((n: any) => (
            <div key={n.id} onClick={() => !n.read && read.mutate(n.id)} className={cx("flex cursor-pointer gap-3 rounded-xl px-3 py-3 transition hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]", !n.read && "bg-indigo-500/[0.04]")}>
              <span className={cx("mt-1.5 h-2 w-2 shrink-0 rounded-full", n.read ? "bg-slate-400/50" : "bg-indigo-500")} />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2"><span className="text-sm font-semibold">{n.title}</span><Badge tone="slate">{n.event}</Badge><Badge tone={n.source === "n8n" ? "fuchsia" : "indigo"}>{n.source}</Badge><span className="muted text-xs">{ago(n.created_at)}</span></div>
                {n.message && <div className="muted mt-1 whitespace-pre-wrap text-xs leading-relaxed">{n.message}</div>}
                {n.payload?.application_id && <Link onClick={(e) => e.stopPropagation()} to={`/pipeline/${n.payload.application_id}`} className="mt-1.5 inline-flex items-center gap-1 text-xs text-indigo-500 hover:underline"><Workflow className="h-3 w-3" /> Open the application</Link>}
              </div>
            </div>
          ))}
        </Card>
        <div className="space-y-5">
          <Card>
            <CardHeader title="n8n orchestrator" subtitle={o.url} icon={<Workflow className="h-4 w-4" />} actions={<Badge tone={o.reachable ? "emerald" : "rose"} dot>{o.reachable ? "reachable" : "not reachable"}</Badge>} />
            <div className="space-y-2 p-5">
              {!o.reachable && <Callout tone="amber">Start it: <code>docker compose up -d n8n</code>, then publish the workflows: <code>python scripts/n8n_setup.py</code>. Without n8n, the API runs the same steps.</Callout>}
              {(o.webhooks || []).map((w: any) => (
                <div key={w.path} className="flex items-center gap-3 rounded-xl border hairline px-3 py-2 text-sm"><Dot tone={w.live ? "emerald" : w.live === false ? "rose" : "slate"} /><div className="min-w-0 flex-1"><div className="truncate font-medium">{w.workflow}</div><div className="muted font-mono text-[11px]">{w.path}</div></div><span className="muted text-xs">{w.live ? "live" : w.live === false ? "not published" : "?"}</span></div>
              ))}
              {o.reachable && !o.api_key && <div className="muted text-xs">Set N8N_API_KEY in .env (n8n → Settings → n8n API) to list the executions here.</div>}
            </div>
          </Card>
          {!!o.executions?.length && (
            <Card>
              <CardHeader title="Latest n8n executions" icon={<Workflow className="h-4 w-4" />} />
              <div className="space-y-1 p-3">
                {o.executions.map((e: any) => (
                  <a key={e.id} href={e.link} target="_blank" rel="noreferrer" className="flex items-center gap-3 rounded-xl px-3 py-2 text-xs hover:bg-slate-900/[0.03] dark:hover:bg-white/[0.04]">
                    <Badge tone={e.status === "success" ? "emerald" : e.status === "waiting" ? "amber" : e.status === "running" ? "sky" : "rose"}>{e.status}</Badge>
                    <span className="min-w-0 flex-1 truncate">{e.workflow}</span><span className="muted">{secs(e.seconds)}</span><span className="muted">{ago(e.started)}</span>
                  </a>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
