import { CheckCircle2, Lock, Send, ShieldCheck, Timer } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { SubmitCv } from "../components/common";
import { Callout, Card, PageHeader } from "../components/ui";

export default function Portal() {
  const [done, setDone] = useState<{ id: string; message?: string } | null>(null);
  return (
    <div>
      <PageHeader eyebrow="Diagram part 1 · what a candidate sees" title="Candidate portal" icon={<Send className="h-7 w-7" />}
        subtitle="A candidate applies here: the CV goes to the application pipeline (through the n8n workflow, or the API). They are e-mailed an acknowledgement, and an answer once a recruiter has decided." />
      <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
        <Card className="overflow-hidden">
          <div className="bg-gradient-to-r from-indigo-500 via-violet-500 to-cyan-500 px-6 py-8 text-white">
            <div className="font-display text-2xl font-semibold">Join our team</div>
            <div className="mt-1 text-sm opacity-90">Send your CV for one of our open positions, or a spontaneous application.</div>
          </div>
          <div className="p-6">
            {done ? (
              <div className="py-6 text-center">
                <CheckCircle2 className="mx-auto h-12 w-12 text-emerald-500" />
                <div className="mt-3 font-display text-xl font-semibold">Application received</div>
                <p className="muted mx-auto mt-2 max-w-md text-sm">{done.message || "A recruiter will answer within 10 working days."}</p>
                <div className="mt-5 flex justify-center gap-2"><button className="btn-ghost" onClick={() => setDone(null)}>Send another</button><Link to={`/pipeline/${done.id}`} className="btn-primary">Follow it (recruiter view)</Link></div>
              </div>
            ) : <SubmitCv source="portal" portal onSubmitted={(id, message) => setDone({ id, message })} />}
          </div>
        </Card>
        <div className="space-y-4">
          <Callout tone="indigo" icon={<ShieldCheck className="h-4 w-4" />} title="A human decides">AI reads and analyses the CV; a recruiter makes every decision (KB-RR-001). Protected attributes are never used (KB-HR-003).</Callout>
          <Callout tone="emerald" icon={<Timer className="h-4 w-4" />} title="An answer within 10 working days">Shortlisted candidates are contacted within 5 working days; interviews come with a video meeting link and a calendar invitation.</Callout>
          <Callout tone="slate" icon={<Lock className="h-4 w-4" />} title="Your data">The file is read once and deleted; the extracted profile is kept at most 24 months. You can ask for access, deletion or a human re-examination at any time (KB-HR-004).</Callout>
        </div>
      </div>
    </div>
  );
}
