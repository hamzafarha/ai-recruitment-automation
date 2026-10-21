// Shared chart styling (Recharts): one palette, readable in both themes.
export const PALETTE = ["#6366f1", "#22d3ee", "#a855f7", "#10b981", "#f59e0b", "#f43f5e", "#0ea5e9", "#d946ef"];
export const BEFORE = "#94a3b8";
export const AFTER = "#6366f1";

export const tooltipStyle = {
  contentStyle: {
    background: "rgba(15, 20, 38, 0.92)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12,
    color: "#e7eaf3", fontSize: 12, boxShadow: "0 12px 30px -12px rgba(0,0,0,0.6)",
  },
  itemStyle: { color: "#e7eaf3" },
  labelStyle: { color: "#a5b4fc", fontWeight: 600 },
  cursor: { fill: "rgba(99,102,241,0.08)" },
};

export const axisProps = { stroke: "currentColor", strokeOpacity: 0.35, tick: { fontSize: 11, fill: "currentColor", fillOpacity: 0.65 }, tickLine: false };
