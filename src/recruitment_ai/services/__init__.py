"""Application services: the use cases behind the API, the interface and n8n.

- ``pipeline``: the application pipeline, ONE workflow from the CV received to the final decision
  (document processing, fine-tuned extraction, storage, screening agent, report, notifications,
  human validation, answer to the candidate), run step by step by n8n or by the API;
- ``candidates``: CV upload -> extraction -> storage -> search index;
- ``jobs``: job creation (or extraction from a job ad) -> storage -> search index;
- ``matching``: fast ranking of candidates for a job, stored evidence-backed analyses;
- ``reviews``: the recruiter's decision on an analysis (human validation);
- ``events``: notifications (in the app and forwarded to n8n);
- ``dashboard``: figures of the dashboard;
- ``demo``: setup (HR knowledge index) and demo data;
- ``container``: the configured models and stores.
"""

from recruitment_ai.services.container import Container, get_container

__all__ = ["Container", "get_container"]
