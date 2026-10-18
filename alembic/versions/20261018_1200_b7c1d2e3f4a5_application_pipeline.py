"""application pipeline: applications table, analyses.application_id

One row per application going through the recruitment pipeline (services/pipeline.py), from the
CV received to the final decision, with the log of its steps. An analysis made by the pipeline's
screening agent points to its application.

Revision ID: b7c1d2e3f4a5
Revises: a1b2c3d4e5f6
Create Date: 2026-10-18 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7c1d2e3f4a5"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "applications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=True,
                  index=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("filename", sa.String(255), nullable=True),
        sa.Column("orchestrator", sa.String(20), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("stage", sa.String(30), nullable=False),
        sa.Column("steps", sa.JSON(), nullable=False),
        sa.Column("work", sa.JSON(), nullable=True),
        sa.Column("decision", sa.String(30), nullable=True),
        sa.Column("resume_url", sa.String(1000), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    # batch mode: a plain ALTER TABLE on PostgreSQL, a table copy on SQLite
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(sa.Column("application_id", sa.String(36), nullable=True))
        batch.create_index("ix_analyses_application_id", ["application_id"])
        batch.create_foreign_key("fk_analyses_application_id", "applications", ["application_id"], ["id"],
                                 ondelete="SET NULL")


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_constraint("fk_analyses_application_id", type_="foreignkey")
        batch.drop_index("ix_analyses_application_id")
        batch.drop_column("application_id")
    op.drop_table("applications")
