"""Application services: the use cases behind the API, the interface and n8n.

- ``pipeline``: the application pipeline, ONE workflow from the CV received to the final decision
  (document processing, fine-tuned extraction, storage, screening agent, report, notifications,
  human validation, answer to the candidate), run step by step by n8n or by the API;
- ``candidates``: CV upload -> extraction -> storage -> search index;
- ``jobs``: job creation (or extraction from a job ad) -> storage -> search index;
- ``matching``: fast ranking of candidates for a job, stored evidence-backed analyses;
- ``reviews``: the recruiter's decision on an analysis (human validation);
- ``events``: notifications (in the app and forwarded to n8n);
- ``mailer`` / ``meetings``: e-mails to the candidates (SMTP or outbox) and interview meetings (Google Meet
  through Google Calendar, or Jitsi) with their calendar invite;
- ``outputs``: what every step of an application produced, in full (the Pipeline page);
- ``knowledge``: the HR knowledge-base documents the RAG retrieves from;
- ``evaluation``: every evaluation result of data/evaluation/, the live pipeline's timings, MLflow status;
- ``observe``: read-only views of each part of the architecture ("Under the hood");
- ``dashboard``: figures of the dashboard;
- ``demo``: setup (HR knowledge index) and demo data;
- ``container``: the configured models and stores.
"""

from recruitment_ai.services.container import Container, get_container

__all__ = ["Container", "get_container"]
