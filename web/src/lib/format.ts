export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const pts = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)} pts`;

export const num = (x: number | null | undefined, digits = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(digits).replace(/\.?0+$/, "");

export const secs = (x: number | null | undefined) => {
  if (x === null || x === undefined) return "—";
  if (x < 1) return `${Math.round(x * 1000)} ms`;
  if (x < 90) return `${num(x, 1)} s`;
  if (x < 5400) return `${Math.round(x / 60)} min`;
  return `${num(x / 3600, 1)} h`;
};

export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export const shortId = (id?: string | null) => (id || "").slice(0, 8);

export const GRADE_LABELS: Record<number, string> = { 3: "Strong fit", 2: "Partial fit", 1: "Weak fit", 0: "No fit" };
export const GRADE_TONE: Record<number, string> = { 3: "emerald", 2: "sky", 1: "amber", 0: "rose" };

export const ANALYSIS_STATUS: Record<string, { label: string; tone: string }> = {
  pending_review: { label: "Pending review", tone: "amber" },
  shortlisted: { label: "Shortlisted", tone: "emerald" },
  interview: { label: "Interview", tone: "violet" },
  on_hold: { label: "On hold", tone: "slate" },
  rejected: { label: "Rejected", tone: "rose" },
};

export const APP_STATUS: Record<string, { label: string; tone: string }> = {
  running: { label: "Running", tone: "sky" },
  waiting_review: { label: "Waiting for a decision", tone: "amber" },
  closed: { label: "Closed", tone: "emerald" },
  failed: { label: "Stopped at a failed step", tone: "rose" },
};

export const DECISIONS: Record<string, { label: string; tone: string }> = {
  interview: { label: "Interview", tone: "violet" },
  shortlist: { label: "Shortlisted", tone: "emerald" },
  reject: { label: "Rejected", tone: "rose" },
  hold: { label: "On hold", tone: "slate" },
};

export const REQ_STATUS: Record<string, { label: string; tone: string }> = {
  met: { label: "Met", tone: "emerald" },
  partial: { label: "Partial", tone: "amber" },
  missing: { label: "Missing", tone: "rose" },
};

export const LEVELS: Record<string, string> = {
  demonstrated: "demonstrated",
  equivalent: "equivalent",
  listed: "listed only",
  not_evidenced: "not evidenced",
};

export const METHOD_LABEL: Record<string, string> = {
  agent: "Screening agent",
  llm: "AI analysis",
  semantic: "Semantic estimate",
};

export const EMAIL_STATUS: Record<string, { label: string; tone: string }> = {
  sent: { label: "Sent", tone: "emerald" },
  outbox: { label: "In the outbox", tone: "sky" },
  failed: { label: "Failed", tone: "rose" },
  skipped: { label: "Not sent", tone: "slate" },
};

export function skillName(s: any): string {
  return (typeof s === "string" ? s : s?.name) || "";
}
