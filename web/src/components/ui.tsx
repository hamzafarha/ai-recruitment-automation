import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, Check, ChevronDown, Copy, FileUp, Info, Loader2, X } from "lucide-react";
import { ReactNode, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { errorText } from "../lib/api";
import { GRADE_LABELS, GRADE_TONE } from "../lib/format";

export const cx = clsx;

// ------------------------------------------------------------------ tones

export const TONES: Record<string, { badge: string; dot: string; soft: string; text: string; ring: string; solid: string }> = {
  emerald: { badge: "bg-emerald-500/10 text-emerald-700 border-emerald-500/25 dark:text-emerald-300", dot: "bg-emerald-500", soft: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-300", ring: "ring-emerald-500/40", solid: "#10b981" },
  sky: { badge: "bg-sky-500/10 text-sky-700 border-sky-500/25 dark:text-sky-300", dot: "bg-sky-500", soft: "bg-sky-500/10", text: "text-sky-600 dark:text-sky-300", ring: "ring-sky-500/40", solid: "#0ea5e9" },
  amber: { badge: "bg-amber-500/10 text-amber-700 border-amber-500/25 dark:text-amber-300", dot: "bg-amber-500", soft: "bg-amber-500/10", text: "text-amber-600 dark:text-amber-300", ring: "ring-amber-500/40", solid: "#f59e0b" },
  rose: { badge: "bg-rose-500/10 text-rose-700 border-rose-500/25 dark:text-rose-300", dot: "bg-rose-500", soft: "bg-rose-500/10", text: "text-rose-600 dark:text-rose-300", ring: "ring-rose-500/40", solid: "#f43f5e" },
  violet: { badge: "bg-violet-500/10 text-violet-700 border-violet-500/25 dark:text-violet-300", dot: "bg-violet-500", soft: "bg-violet-500/10", text: "text-violet-600 dark:text-violet-300", ring: "ring-violet-500/40", solid: "#8b5cf6" },
  indigo: { badge: "bg-indigo-500/10 text-indigo-700 border-indigo-500/25 dark:text-indigo-300", dot: "bg-indigo-500", soft: "bg-indigo-500/10", text: "text-indigo-600 dark:text-indigo-300", ring: "ring-indigo-500/40", solid: "#6366f1" },
  cyan: { badge: "bg-cyan-500/10 text-cyan-700 border-cyan-500/25 dark:text-cyan-300", dot: "bg-cyan-500", soft: "bg-cyan-500/10", text: "text-cyan-600 dark:text-cyan-300", ring: "ring-cyan-500/40", solid: "#06b6d4" },
  fuchsia: { badge: "bg-fuchsia-500/10 text-fuchsia-700 border-fuchsia-500/25 dark:text-fuchsia-300", dot: "bg-fuchsia-500", soft: "bg-fuchsia-500/10", text: "text-fuchsia-600 dark:text-fuchsia-300", ring: "ring-fuchsia-500/40", solid: "#d946ef" },
  slate: { badge: "bg-slate-500/10 text-slate-600 border-slate-500/20 dark:text-slate-300", dot: "bg-slate-400", soft: "bg-slate-500/10", text: "text-slate-500 dark:text-slate-300", ring: "ring-slate-500/30", solid: "#94a3b8" },
};
const tone = (t?: string) => TONES[t || "slate"] || TONES.slate;

// ------------------------------------------------------------------ layout blocks

export function Card({ className, children, hover, onClick }: { className?: string; children: ReactNode; hover?: boolean; onClick?: () => void }) {
  return (
    <div onClick={onClick} className={cx("card", hover && "card-hover cursor-pointer", className)}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, icon, actions, className }: { title: ReactNode; subtitle?: ReactNode; icon?: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <div className={cx("flex items-start justify-between gap-4 px-5 pt-5", className)}>
      <div className="flex min-w-0 items-start gap-3">
        {icon && <IconChip>{icon}</IconChip>}
        <div className="min-w-0">
          <h3 className="font-display text-[15px] font-semibold leading-tight">{title}</h3>
          {subtitle && <p className="muted mt-1 text-xs leading-relaxed">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

export function IconChip({ children, tone: t = "indigo", className }: { children: ReactNode; tone?: string; className?: string }) {
  return (
    <span className={cx("grid h-9 w-9 shrink-0 place-items-center rounded-xl border", tone(t).badge, className)}>
      {children}
    </span>
  );
}

export function PageHeader({ title, subtitle, icon, actions, eyebrow }: { title: ReactNode; subtitle?: ReactNode; icon?: ReactNode; actions?: ReactNode; eyebrow?: string }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 max-w-3xl">
        {eyebrow && <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-500 dark:text-indigo-300">{eyebrow}</div>}
        <h1 className="flex items-center gap-3 font-display text-[28px] font-semibold leading-tight tracking-tight">
          {icon && <span className="text-indigo-500 dark:text-indigo-300">{icon}</span>}
          <span className="gradient-text">{title}</span>
        </h1>
        {subtitle && <p className="muted mt-2 text-sm leading-relaxed">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Section({ title, desc, actions, children, className }: { title: ReactNode; desc?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cx("mb-8", className)}>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-semibold">{title}</h2>
          {desc && <p className="muted mt-0.5 text-sm">{desc}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

// ------------------------------------------------------------------ small pieces

export function Badge({ tone: t, children, dot, className, title }: { tone?: string; children: ReactNode; dot?: boolean; className?: string; title?: string }) {
  return (
    <span title={title} className={cx("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold", tone(t).badge, className)}>
      {dot && <span className={cx("h-1.5 w-1.5 rounded-full", tone(t).dot)} />}
      {children}
    </span>
  );
}

export function Dot({ tone: t, pulse }: { tone?: string; pulse?: boolean }) {
  return (
    <span className="relative inline-flex h-2 w-2">
      {pulse && <span className={cx("absolute inline-flex h-full w-full animate-ping rounded-full opacity-60", tone(t).dot)} />}
      <span className={cx("relative inline-flex h-2 w-2 rounded-full", tone(t).dot)} />
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cx("h-4 w-4 animate-spin", className)} />;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("skeleton h-4", className)} />;
}

export function Loading({ label = "Loading…", rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div className="space-y-3 p-1">
      <div className="muted flex items-center gap-2 text-sm"><Spinner /> {label}</div>
      {Array.from({ length: rows }).map((_, i) => <Skeleton key={i} className={i % 2 ? "w-2/3" : "w-full"} />)}
    </div>
  );
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: ReactNode; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      {icon && <div className="mb-4 grid h-14 w-14 place-items-center rounded-2xl border hairline surface-2 text-indigo-500 dark:text-indigo-300">{icon}</div>}
      <div className="font-display text-base font-semibold">{title}</div>
      {children && <div className="muted mt-1.5 max-w-md text-sm leading-relaxed">{children}</div>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Callout({ tone: t = "indigo", title, children, icon, className }: { tone?: string; title?: ReactNode; children?: ReactNode; icon?: ReactNode; className?: string }) {
  return (
    <div className={cx("flex gap-3 rounded-xl border px-4 py-3 text-sm", tone(t).badge, className)}>
      <div className="mt-0.5 shrink-0">{icon ?? (t === "rose" || t === "amber" ? <AlertTriangle className="h-4 w-4" /> : <Info className="h-4 w-4" />)}</div>
      <div className="min-w-0 leading-relaxed">
        {title && <div className="font-semibold">{title}</div>}
        {children && <div className={cx(title && "mt-0.5", "opacity-90")}>{children}</div>}
      </div>
    </div>
  );
}

export function ErrorBox({ error, className }: { error: unknown; className?: string }) {
  if (!error) return null;
  return <Callout tone="rose" title="Something went wrong" className={className}>{errorText(error)}</Callout>;
}

export function Stat({ label, value, hint, icon, tone: t = "indigo", className }: { label: ReactNode; value: ReactNode; hint?: ReactNode; icon?: ReactNode; tone?: string; className?: string }) {
  return (
    <Card className={cx("relative overflow-hidden p-5", className)}>
      <div className={cx("pointer-events-none absolute -right-8 -top-8 h-28 w-28 rounded-full blur-2xl", tone(t).soft)} />
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="muted text-xs font-medium uppercase tracking-wider">{label}</div>
          <div className="mt-2 font-display text-[28px] font-semibold leading-none tracking-tight">{value}</div>
          {hint && <div className="muted mt-2 text-xs">{hint}</div>}
        </div>
        {icon && <IconChip tone={t}>{icon}</IconChip>}
      </div>
    </Card>
  );
}

export function KeyVals({ items, cols = 2 }: { items: [ReactNode, ReactNode][]; cols?: number }) {
  return (
    <dl className={cx("grid gap-x-6 gap-y-3 text-sm", cols === 3 ? "sm:grid-cols-3" : cols === 4 ? "sm:grid-cols-4" : "sm:grid-cols-2")}>
      {items.map(([k, v], i) => (
        <div key={i} className="min-w-0">
          <dt className="muted text-[11px] font-medium uppercase tracking-wider">{k}</dt>
          <dd className="mt-0.5 break-words font-medium">{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Progress({ value, tone: t = "indigo", className }: { value: number; tone?: string; className?: string }) {
  return (
    <div className={cx("h-1.5 w-full overflow-hidden rounded-full bg-slate-500/15", className)}>
      <motion.div initial={{ width: 0 }} animate={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} transition={{ duration: 0.8, ease: "easeOut" }}
        className="h-full rounded-full" style={{ background: tone(t).solid }} />
    </div>
  );
}

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={cx("block", className)}>
      <span className="label">{label}</span>
      {children}
      {hint && <span className="muted mt-1 block text-xs">{hint}</span>}
    </label>
  );
}

// ------------------------------------------------------------------ tabs, collapsible, drawer

export function Tabs<T extends string>({ tabs, value, onChange, className }: { tabs: { id: T; label: ReactNode; icon?: ReactNode; count?: number }[]; value: T; onChange: (v: T) => void; className?: string }) {
  return (
    <div className={cx("mb-5 flex gap-1 overflow-x-auto rounded-2xl border hairline surface-2 p-1", className)}>
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={cx("relative flex shrink-0 items-center gap-2 rounded-xl px-3.5 py-2 text-sm font-medium transition",
            value === t.id ? "text-[color:var(--text)]" : "muted hover:text-[color:var(--text)]")}>
          {value === t.id && <motion.span layoutId={`tab-${tabs.map((x) => x.id).join("")}`} className="absolute inset-0 rounded-xl border hairline bg-white shadow-sm dark:bg-white/[0.08]" transition={{ type: "spring", bounce: 0.18, duration: 0.45 }} />}
          <span className="relative flex items-center gap-2">{t.icon}{t.label}{t.count !== undefined && <span className="rounded-md bg-slate-500/15 px-1.5 text-[10px]">{t.count}</span>}</span>
        </button>
      ))}
    </div>
  );
}

export function Collapsible({ title, right, children, defaultOpen = false, className }: { title: ReactNode; right?: ReactNode; children: ReactNode; defaultOpen?: boolean; className?: string }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={cx("rounded-xl border hairline", className)}>
      <button onClick={() => setOpen(!open)} className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm font-medium">
        <span className="flex min-w-0 items-center gap-2">{title}</span>
        <span className="flex items-center gap-2">{right}<ChevronDown className={cx("h-4 w-4 transition", open && "rotate-180")} /></span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="border-t hairline px-4 py-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function Drawer({ open, onClose, title, subtitle, children, width = "max-w-3xl", actions }: { open: boolean; onClose: () => void; title: ReactNode; subtitle?: ReactNode; children: ReactNode; width?: string; actions?: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} className="absolute inset-0 bg-slate-950/50 backdrop-blur-sm" />
          <motion.aside initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ type: "spring", damping: 30, stiffness: 280 }}
            className={cx("absolute right-0 top-0 flex h-full w-full flex-col border-l hairline bg-[color:var(--bg)] shadow-2xl", width)}>
            <div className="flex items-start justify-between gap-4 border-b hairline px-6 py-4">
              <div className="min-w-0">
                <div className="font-display text-lg font-semibold">{title}</div>
                {subtitle && <div className="muted mt-0.5 text-xs">{subtitle}</div>}
              </div>
              <div className="flex items-center gap-2">
                {actions}
                <button onClick={onClose} className="btn-ghost btn-sm"><X className="h-4 w-4" /></button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-6 py-5">{children}</div>
          </motion.aside>
        </div>
      )}
    </AnimatePresence>
  );
}

// ------------------------------------------------------------------ code, json, markdown

export function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button className="btn-ghost btn-sm" onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1400); }}>
      {done ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />} {done ? "Copied" : "Copy"}
    </button>
  );
}

export function CodeBlock({ code, title, maxHeight = "max-h-96", className }: { code: string; title?: ReactNode; maxHeight?: string; className?: string }) {
  return (
    <div className={cx("overflow-hidden rounded-xl border hairline bg-slate-950 text-slate-100 dark:bg-black/40", className)}>
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-1.5">
        <span className="font-mono text-[11px] text-slate-400">{title}</span>
        <CopyButton text={code} />
      </div>
      <pre className={cx("overflow-auto p-3.5 font-mono text-[12px] leading-relaxed whitespace-pre-wrap break-words", maxHeight)}>{code}</pre>
    </div>
  );
}

export function JsonView({ data, title = "JSON", maxHeight }: { data: unknown; title?: string; maxHeight?: string }) {
  return <CodeBlock code={JSON.stringify(data, null, 2)} title={title} maxHeight={maxHeight} />;
}

export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <div className={cx("prose-x", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

// ------------------------------------------------------------------ grades

export function GradeBadge({ grade, label }: { grade?: number | null; label?: string | null }) {
  if (grade === null || grade === undefined) return <span className="muted">—</span>;
  return <Badge tone={GRADE_TONE[grade]} dot>Grade {grade} · {label || GRADE_LABELS[grade]}</Badge>;
}

export function GradeRing({ score, grade, size = 96 }: { score: number; grade?: number; size?: number }) {
  const r = (size - 10) / 2;
  const c = 2 * Math.PI * r;
  const color = TONES[GRADE_TONE[grade ?? 0] || "slate"].solid;
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="currentColor" strokeOpacity={0.12} strokeWidth={8} fill="none" />
        <motion.circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={8} fill="none" strokeLinecap="round"
          strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - Math.min(100, score) / 100) }} transition={{ duration: 1.1, ease: "easeOut" }} />
      </svg>
      <div className="absolute text-center">
        <div className="font-display text-2xl font-semibold leading-none">{Math.round(score)}</div>
        <div className="muted mt-0.5 text-[10px] uppercase tracking-wider">/ 100</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ file drop

export function Dropzone({ file, onFile, accept = ".pdf,.docx,.png,.jpg,.jpeg", hint = "PDF (digital or scanned), Word or image · max 15 MB" }: { file: File | null; onFile: (f: File | null) => void; accept?: string; hint?: string }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
      className={cx("group flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-8 text-center transition",
        over ? "border-indigo-400 bg-indigo-500/10" : "hairline hover:border-indigo-400/60 hover:bg-indigo-500/[0.04]")}>
      <input ref={input} type="file" accept={accept} className="hidden" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
      <div className="mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-cyan-400 text-white shadow-lg shadow-indigo-500/30 transition group-hover:scale-105">
        <FileUp className="h-5 w-5" />
      </div>
      {file ? (
        <div className="text-sm"><span className="font-semibold">{file.name}</span> <span className="muted">· {(file.size / 1024).toFixed(0)} KB</span></div>
      ) : (
        <div className="text-sm font-medium">Drop a CV here, or click to choose a file</div>
      )}
      <div className="muted mt-1 text-xs">{hint}</div>
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <button type="button" onClick={() => onChange(!checked)} className="flex items-center gap-3 text-left text-sm">
      <span className={cx("relative h-5 w-9 shrink-0 rounded-full transition", checked ? "bg-indigo-500" : "bg-slate-400/40")}>
        <span className={cx("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition", checked ? "left-[18px]" : "left-0.5")} />
      </span>
      <span>{label}</span>
    </button>
  );
}
