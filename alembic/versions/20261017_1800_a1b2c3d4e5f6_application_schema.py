"""application schema: candidates, jobs, analyses, reviews, notifications, agent runs

Replaces the initial scaffold tables (never used: they split a profile over ten tables) with the
tables the application works with. Profiles, requirements and reports are stored as JSON.

Revision ID: a1b2c3d4e5f6
Revises: 775984690766
Create Date: 2026-10-17 18:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "775984690766"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# children first, so foreign keys never block a drop
SCAFFOLD_TABLES = ["recruiter_feedbacks", "candidate_matches", "analyses", "skills", "projects", "languages",
                   "job_requirements", "experiences", "educations", "certifications", "candidate_documents",
                   "jobs", "candidates"]
NEW_TABLES = ["agent_runs", "notifications", "reviews", "analyses", "jobs", "candidates"]


def _drop_existing(names: list[str]) -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for name in names:
        if name in existing:
            op.drop_table(name)


def upgrade() -> None:
    _drop_existing(SCAFFOLD_TABLES)
    op.create_table(
        "candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("external_id", sa.String(100), nullable=True, index=True),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("email", sa.String(255), nullable=True, index=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("profile", sa.JSON(), nullable=False),
        sa.Column("cv_text", sa.Text(), nullable=True),
        sa.Column("cv_filename", sa.String(255), nullable=True),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("extraction", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("external_id", sa.String(100), nullable=True, index=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("company", sa.String(255), nullable=True),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("employment_type", sa.String(100), nullable=True),
        sa.Column("role", sa.String(80), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("requirements", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
                  index=True),
        sa.Column("method", sa.String(20), nullable=False),
        sa.Column("grade", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("must_coverage", sa.Float(), nullable=True),
        sa.Column("nice_coverage", sa.Float(), nullable=True),
        sa.Column("action", sa.String(30), nullable=True),
        sa.Column("report", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("final_grade", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False,
                  index=True),
        sa.Column("reviewer", sa.String(100), nullable=False),
        sa.Column("decision", sa.String(30), nullable=False),
        sa.Column("adjusted_grade", sa.Integer(), nullable=True),
        sa.Column("interview_at", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "notifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("event", sa.String(60), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("read", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
    )
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("steps", sa.JSON(), nullable=True),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    # The scaffold tables were never used; going back only removes the application tables.
    _drop_existing(NEW_TABLES)
