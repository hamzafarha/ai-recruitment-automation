import { useQuery } from "@tanstack/react-query";
import { Database as DbIcon } from "lucide-react";
import { useState } from "react";
import { ErrorState } from "../components/common";
import { Badge, Card, cx, JsonView, Loading, PageHeader } from "../components/ui";
import { api } from "../lib/api";

function Cell({ value }: { value: any }) {
  const [open, setOpen] = useState(false);
  if (value === null || value === undefined) return <span className="muted">∅</span>;
  if (typeof value === "object") {
    return open ? <div className="min-w-[320px]"><button className="text-[10px] text-indigo-500" onClick={() => setOpen(false)}>hide</button><JsonView data={value} maxHeight="max-h-64" /></div>
      : <button className="chip text-[10px] hover:border-indigo-400" onClick={() => setOpen(true)}>{Array.isArray(value) ? `[${value.length}]` : `{${Object.keys(value).length}}`} JSON</button>;
  }
  const text = String(value);
  return <span className="block max-w-[320px] truncate" title={text}>{text}</span>;
}

export default function DatabasePage() {
  const [name, setName] = useState("applications");
  const [offset, setOffset] = useState(0);
  const tables = useQuery({ queryKey: ["tables"], queryFn: () => api.get("/observe/tables") });
  const rows = useQuery({ queryKey: ["table", name, offset], queryFn: () => api.get(`/observe/tables/${name}?limit=25&offset=${offset}`) });
  return (
    <div>
      <PageHeader eyebrow="Diagram part 8" title="Database (PostgreSQL)" icon={<DbIcon className="h-7 w-7" />} subtitle={<>Every table of the application, newest rows first; JSON columns open in place. Migration: <code>{tables.data?.migration || "—"}</code>.</>} />
      {tables.error ? <ErrorState error={tables.error} /> : (
        <div className="mb-5 flex flex-wrap gap-2">
          {(tables.data?.tables || []).map((t: any) => (
            <button key={t.table} onClick={() => { setName(t.table); setOffset(0); }} className={cx("chip px-3 py-1.5", t.table === name && "border-indigo-400 bg-indigo-500/10 text-indigo-700 dark:text-indigo-200")}>{t.table}<Badge tone="slate">{t.rows}</Badge></button>
          ))}
        </div>
      )}
      <Card className="overflow-hidden">
        {rows.isLoading ? <div className="p-5"><Loading /></div> : rows.error ? <ErrorState error={rows.error} /> : (
          <>
            <div className="overflow-x-auto">
              <table className="table-x text-xs">
                <thead><tr>{rows.data.columns.map((c: any) => <th key={c.name} title={c.type}>{c.name}</th>)}</tr></thead>
                <tbody>{rows.data.rows.map((r: any, i: number) => <tr key={i}>{rows.data.columns.map((c: any) => <td key={c.name}><Cell value={r[c.name]} /></td>)}</tr>)}</tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t hairline px-4 py-3 text-xs">
              <span className="muted">{rows.data.total} rows · showing {offset + 1}-{Math.min(offset + 25, rows.data.total)}</span>
              <div className="flex gap-2"><button className="btn-ghost btn-sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}>Previous</button><button className="btn-ghost btn-sm" disabled={offset + 25 >= rows.data.total} onClick={() => setOffset(offset + 25)}>Next</button></div>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
