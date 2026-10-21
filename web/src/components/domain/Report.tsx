// The evidence-backed report of an analysis (diagram part 6), with the HR rules it used (RAG).
import { BookOpen, Download, FileText, HelpCircle, ShieldCheck, TrendingDown, TrendingUp } from "lucide-react";
import { API_PREFIX } from "../../lib/api";
import { ANALYSIS_STATUS, LEVELS, METHOD_LABEL, pct, REQ_STATUS } from "../../lib/format";
import { Badge, Callout, Collapsible, cx, GradeBadge, GradeRing } from "../ui";
import { DocLink } from "./DocViewer";

export function RequirementsTable({ requirements }: { requirements: any[] }) {
  return (
    <div className="overflow-x-auto rounded-2xl border hairline">
      <table className="table-x">
        <thead><tr><th>Req.</th><th>Requirement</th><th>Status</th><th>Evidence level</th><th className="min-w-[280px]">CV evidence</th></tr></thead>
        <tbody>
          {requirements.map((r) => (
            <tr key={r.id}>
              <td><Badge tone={r.kind === "must" ? "indigo" : "slate"}>{r.id}</Badge><div className="muted mt-1 text-[10px] uppercase">{r.kind === "must" ? "must" : "nice"} · {r.importance}</div></td>
              <td className="max-w-[260px]">{r.text}{r.min_years ? <div className="muted mt-0.5 text-xs">min. {r.min_years} years</div> : null}</td>
              <td><Badge tone={REQ_STATUS[r.status]?.tone} dot>{REQ_STATUS[r.status]?.label || r.status}</Badge></td>
              <td className="text-xs">
                {LEVELS[r.level] || r.level}
                {r.min_years && r.years_found !== null && r.years_found !== undefined ? <div className="muted">{r.years_found} y / {r.min_years} y</div> : null}
                {r.confidence !== null && r.confidence !== undefined ? <div className="muted">relevance {Number(r.confidence).toFixed(2)}</div> : null}
              </td>
              <td className="text-[13px]">
                {r.evidence ? (
                  <>
                    <div className="rounded-lg border-l-2 border-indigo-400/70 bg-indigo-500/[0.05] px-2.5 py-1.5 italic">“{r.evidence}”</div>
                    <div className="muted mt-1 text-[11px]">{r.evidence_source}{r.grounded === false && <span className="ml-1 text-amber-600"> · not found in the CV</span>}</div>
                  </>
                ) : <span className="muted">—</span>}
                {r.note && <div className="mt-1 text-[11px] text-violet-600 dark:text-violet-300">check: {r.note}</div>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function KnowledgeSources({ sources }: { sources: any[] }) {
  if (!sources?.length) return <div className="muted text-sm">No HR rule was retrieved.</div>;
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {sources.map((k, i) => (
        <div key={i} className="rounded-xl border hairline p-3">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
            <DocLink docId={k.doc_id} section={k.section} chunkId={k.chunk_id}>{k.doc_id}</DocLink>
            <span className="font-medium">{k.title}</span>
            {k.section && <span className="muted">· {k.section}</span>}
            {k.score !== null && k.score !== undefined && <Badge tone="cyan">rerank {Number(k.score).toFixed(3)}</Badge>}
          </div>
          <p className="muted line-clamp-4 text-xs leading-relaxed">{k.excerpt}</p>
        </div>
      ))}
    </div>
  );
}

export function ReportView({ analysis }: { analysis: any }) {
  const report = analysis.report || {};
  const final = analysis.final_grade;
  const must = (report.requirements || []).filter((r: any) => r.kind === "must");
  const count = (s: string) => must.filter((r: any) => r.status === s).length;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-6 rounded-2xl border hairline bg-gradient-to-br from-indigo-500/[0.06] via-transparent to-cyan-500/[0.05] p-5">
        <GradeRing score={report.score || 0} grade={report.grade} />
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <GradeBadge grade={report.grade} label={report.grade_label} />
            {final !== null && final !== undefined && final !== report.grade && <Badge tone="violet">adjusted to {final} by the recruiter</Badge>}
            <Badge tone={ANALYSIS_STATUS[analysis.status]?.tone}>{ANALYSIS_STATUS[analysis.status]?.label || analysis.status}</Badge>
            <Badge tone="slate">{METHOD_LABEL[report.method] || report.method}{report.model ? ` · ${report.model.split("/").pop()}` : ""}</Badge>
          </div>
          <div className="text-sm leading-relaxed">{report.summary || "—"}</div>
          <div className="muted flex flex-wrap gap-x-5 gap-y-1 text-xs">
            <span>Must-have coverage <b className="text-[color:var(--text)]">{pct(report.must_coverage)}</b></span>
            <span>Met {count("met")} · partial {count("partial")} · missing {count("missing")}</span>
            <span>Experience <b className="text-[color:var(--text)]">{report.total_years ?? "?"} years</b> (KB-RR-003)</span>
            <span>Proposed: <b className="text-[color:var(--text)]">{report.action_label}</b></span>
          </div>
        </div>
        {analysis.id && (
          <div className="flex flex-col gap-2">
            <a className="btn-ghost btn-sm" href={`${API_PREFIX}/analyses/${analysis.id}/report.pdf`}><Download className="h-3.5 w-3.5" /> PDF</a>
            <a className="btn-ghost btn-sm" href={`${API_PREFIX}/analyses/${analysis.id}/report.md`}><FileText className="h-3.5 w-3.5" /> Markdown</a>
          </div>
        )}
      </div>
      {(report.warnings || []).map((w: string, i: number) => <Callout key={i} tone="amber">{w}</Callout>)}
      <RequirementsTable requirements={report.requirements || []} />
      <div className="grid gap-4 md:grid-cols-3">
        {[["Strengths", report.strengths, <TrendingUp key="s" className="h-4 w-4 text-emerald-500" />], ["Gaps to explore", report.gaps, <TrendingDown key="g" className="h-4 w-4 text-amber-500" />], ["Interview questions", report.interview_questions, <HelpCircle key="q" className="h-4 w-4 text-violet-500" />]].map(([title, items, icon]: any) => (
          <div key={title} className="rounded-2xl border hairline p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">{icon}{title}</div>
            {items?.length ? <ul className="space-y-1.5 text-[13px] leading-relaxed">{items.map((x: string, i: number) => <li key={i} className="flex gap-2"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-current opacity-50" />{x}</li>)}</ul> : <div className="muted text-sm">—</div>}
          </div>
        ))}
      </div>
      <Collapsible defaultOpen title={<><BookOpen className="h-4 w-4 text-indigo-500" /> Company rules used — {report.knowledge_sources?.length || 0} HR knowledge chunks retrieved (RAG)</>}>
        <KnowledgeSources sources={report.knowledge_sources || []} />
      </Collapsible>
      <div className={cx("muted flex items-center gap-2 text-xs")}><ShieldCheck className="h-3.5 w-3.5" /> Decision support only: a recruiter makes every decision (KB-RR-001). “Not evidenced” means the CV does not mention it, not that the candidate lacks it.</div>
    </div>
  );
}
