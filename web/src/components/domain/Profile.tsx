// A structured CV (what the extraction step wrote) and the base vs fine-tuned comparison.
import { Award, Briefcase, FolderGit2, GraduationCap, Languages, Mail, MapPin, Phone, Sparkles, User } from "lucide-react";
import { ReactNode } from "react";
import { skillName } from "../../lib/format";
import { Badge, cx } from "../ui";

function Block({ icon, title, count, children, className }: { icon: ReactNode; title: string; count?: number; children: ReactNode; className?: string }) {
  return (
    <div className={cx("rounded-2xl border hairline p-4", className)}>
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
        <span className="text-indigo-500 dark:text-indigo-300">{icon}</span>{title}
        {count !== undefined && <span className="muted rounded-md bg-slate-500/10 px-1.5 text-[11px] font-medium">{count}</span>}
      </div>
      {children}
    </div>
  );
}

const nothing = <div className="muted text-sm italic">Not found in the CV</div>;
const period = (a?: string | null, b?: string | null, current?: boolean) =>
  [a, b || (current ? "present" : null)].filter(Boolean).join(" → ") || null;

export function ProfileView({ profile, highlight, compact }: { profile: any; highlight?: Set<string>; compact?: boolean }) {
  if (!profile) return <div className="muted text-sm">No profile yet.</div>;
  const skills: any[] = profile.skills || [];
  const groups = skills.reduce((acc: Record<string, any[]>, s: any) => {
    const key = (typeof s === "object" && s?.category) || "Skills";
    (acc[key] ||= []).push(s);
    return acc;
  }, {});
  const hl = (name: string) => highlight?.has(name.trim().toLowerCase());
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start gap-4 rounded-2xl border hairline bg-gradient-to-br from-indigo-500/[0.07] to-cyan-500/[0.04] p-4">
        <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-500 text-lg font-semibold text-white">
          {(profile.full_name || "?").split(" ").map((w: string) => w[0]).slice(0, 2).join("")}
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-display text-lg font-semibold">{profile.full_name || <span className="muted italic">Name not found</span>}</div>
          <div className="muted mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs">
            <span className="flex items-center gap-1"><Mail className="h-3.5 w-3.5" />{profile.email || "—"}</span>
            <span className="flex items-center gap-1"><Phone className="h-3.5 w-3.5" />{profile.phone || "—"}</span>
            <span className="flex items-center gap-1"><MapPin className="h-3.5 w-3.5" />{profile.location || "—"}</span>
          </div>
          {profile.summary && <p className="mt-2 text-sm leading-relaxed">{profile.summary}</p>}
        </div>
      </div>

      <div className={cx("grid gap-4", !compact && "lg:grid-cols-2")}>
        <Block icon={<Briefcase className="h-4 w-4" />} title="Experience" count={(profile.experience || []).length} className={!compact ? "lg:col-span-2" : ""}>
          {(profile.experience || []).length ? (
            <ol className="relative space-y-4 border-l hairline pl-5">
              {profile.experience.map((e: any, i: number) => (
                <li key={i} className="relative">
                  <span className="absolute -left-[26px] top-1.5 h-3 w-3 rounded-full border-2 border-indigo-400 bg-[color:var(--bg)]" />
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div className="font-medium">{e.title || <span className="muted italic">title not found</span>} <span className="muted font-normal">· {e.company || "company not found"}</span></div>
                    <span className="muted font-mono text-[11px]">{period(e.start_date, e.end_date, e.is_current) || "dates not found"}</span>
                  </div>
                  {e.description && <p className="muted mt-1 text-[13px] leading-relaxed">{e.description}</p>}
                </li>
              ))}
            </ol>
          ) : nothing}
        </Block>

        <Block icon={<GraduationCap className="h-4 w-4" />} title="Education" count={(profile.education || []).length}>
          {(profile.education || []).length ? (
            <div className="space-y-3">
              {profile.education.map((e: any, i: number) => (
                <div key={i}>
                  <div className="font-medium">{[e.degree, e.field_of_study].filter(Boolean).join(" · ") || <span className="muted italic">degree not found</span>}</div>
                  <div className="muted text-xs">{e.institution || "institution not found"}{period(e.start_date, e.end_date) ? ` · ${period(e.start_date, e.end_date)}` : ""}{e.grade ? ` · ${e.grade}` : ""}</div>
                </div>
              ))}
            </div>
          ) : nothing}
        </Block>

        <Block icon={<FolderGit2 className="h-4 w-4" />} title="Projects" count={(profile.projects || []).length}>
          {(profile.projects || []).length ? (
            <div className="space-y-3">
              {profile.projects.map((p: any, i: number) => (
                <div key={i}>
                  <div className="font-medium">{p.title}</div>
                  {p.description && <p className="muted mt-0.5 text-[13px] leading-relaxed">{p.description}</p>}
                  {!!(p.technologies || []).length && (
                    <div className="mt-1.5 flex flex-wrap gap-1">{p.technologies.map((t: string) => <Badge key={t} tone="cyan">{t}</Badge>)}</div>
                  )}
                </div>
              ))}
            </div>
          ) : nothing}
        </Block>

        <Block icon={<Sparkles className="h-4 w-4" />} title="Skills" count={skills.length} className={!compact ? "lg:col-span-2" : ""}>
          {skills.length ? (
            <div className="space-y-2.5">
              {Object.entries(groups).map(([group, items]) => (
                <div key={group}>
                  {Object.keys(groups).length > 1 && <div className="muted mb-1 text-[11px] font-semibold uppercase tracking-wider">{group}</div>}
                  <div className="flex flex-wrap gap-1.5">
                    {items.map((s: any, i: number) => (
                      <span key={i} title={[s?.proficiency_level, s?.years_experience ? `${s.years_experience} y` : null].filter(Boolean).join(" · ")}
                        className={cx("chip", hl(skillName(s)) ? "border-amber-400/60 bg-amber-300/15 text-amber-700 dark:text-amber-200" : "")}>
                        {skillName(s)}{s?.proficiency_level ? <span className="muted">· {s.proficiency_level}</span> : null}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : nothing}
        </Block>

        <Block icon={<Award className="h-4 w-4" />} title="Certifications" count={(profile.certifications || []).length}>
          {(profile.certifications || []).length ? (
            <ul className="space-y-1.5 text-sm">
              {profile.certifications.map((c: any, i: number) => (
                <li key={i}><span className="font-medium">{typeof c === "string" ? c : c.name}</span>{c?.issuer ? <span className="muted"> · {c.issuer}</span> : null}{c?.issue_date ? <span className="muted"> · {c.issue_date}</span> : null}</li>
              ))}
            </ul>
          ) : nothing}
        </Block>

        <Block icon={<Languages className="h-4 w-4" />} title="Languages" count={(profile.languages || []).length}>
          {(profile.languages || []).length ? (
            <div className="flex flex-wrap gap-1.5">
              {profile.languages.map((l: any, i: number) => <Badge key={i} tone="violet">{l.language}{l.proficiency ? ` · ${l.proficiency}` : ""}</Badge>)}
            </div>
          ) : nothing}
        </Block>
      </div>
    </div>
  );
}

const SECTIONS = ["experience", "education", "projects", "skills", "certifications", "languages"];

export function ProfileCompare({ result }: { result: any }) {
  if (!result) return null;
  const cmp = result.comparison || {};
  const onlyFt = new Set<string>((cmp.skills_only_finetuned || []).map((s: string) => s.toLowerCase()));
  const onlyBase = new Set<string>((cmp.skills_only_base || []).map((s: string) => s.toLowerCase()));
  const ft = result.finetuned || {};
  const base = result.base || {};
  return (
    <div className="space-y-5">
      <div className="overflow-x-auto rounded-2xl border hairline">
        <table className="table-x">
          <thead>
            <tr><th>Section</th><th>Qwen2.5-7B base</th><th>+ QLoRA adapter (ours)</th><th>Difference</th></tr>
          </thead>
          <tbody>
            {SECTIONS.map((s) => {
              const c = cmp.counts?.[s] || {};
              const d = (c.finetuned ?? 0) - (c.base ?? 0);
              return (
                <tr key={s}>
                  <td className="font-medium capitalize">{s}</td>
                  <td>{c.base ?? "—"}</td>
                  <td>{c.finetuned ?? "—"}</td>
                  <td>{d === 0 ? <span className="muted">same</span> : <Badge tone={d > 0 ? "emerald" : "amber"}>{d > 0 ? "+" : ""}{d}</Badge>}</td>
                </tr>
              );
            })}
            {Object.entries(cmp.identity || {}).map(([field, v]: [string, any]) => (
              <tr key={field}>
                <td className="font-medium">{field.replace("_", " ")}</td>
                <td className="break-all">{v.base ?? <span className="muted">—</span>}</td>
                <td className="break-all">{v.finetuned ?? <span className="muted">—</span>}</td>
                <td>{(v.base ?? null) === (v.finetuned ?? null) ? <span className="muted">same</span> : <Badge tone="violet">differs</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid gap-4 text-xs sm:grid-cols-2">
        <div className="rounded-xl border hairline p-3"><div className="mb-1.5 font-semibold">Skills only the base model found</div>{onlyBase.size ? <div className="flex flex-wrap gap-1">{[...onlyBase].map((s) => <Badge key={s} tone="amber">{s}</Badge>)}</div> : <span className="muted">none</span>}</div>
        <div className="rounded-xl border hairline p-3"><div className="mb-1.5 font-semibold">Skills only the fine-tuned model found</div>{onlyFt.size ? <div className="flex flex-wrap gap-1">{[...onlyFt].map((s) => <Badge key={s} tone="emerald">{s}</Badge>)}</div> : <span className="muted">none</span>}</div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2 text-sm font-semibold">
            <User className="h-4 w-4" /> Base Qwen2.5-7B-Instruct
            <Badge tone="slate">{base.seconds ? `${base.seconds} s` : "—"}</Badge>
            {base.output_tokens && <Badge tone="slate">{base.prompt_tokens} → {base.output_tokens} tokens</Badge>}
          </div>
          <ProfileView profile={base.profile} highlight={onlyBase} compact />
        </div>
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2 text-sm font-semibold">
            <Sparkles className="h-4 w-4 text-indigo-500" /> Qwen2.5-7B + our QLoRA adapter
            <Badge tone="indigo">fine-tuned on 873 CVs</Badge>
            {ft.seconds && <Badge tone="slate">{ft.seconds} s</Badge>}
          </div>
          <ProfileView profile={ft.profile} highlight={onlyFt} compact />
        </div>
      </div>
    </div>
  );
}
