"""Create jobs table and link predictions to jobs.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("idempotency_key", sa.String(255)),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("processed", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(500)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')", name="ck_jobs_status"
        ),
        # NULL keys never conflict, so jobs submitted without a key are not affected.
        sa.UniqueConstraint("user_id", "idempotency_key", name="uq_jobs_user_id_idempotency_key"),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])

    op.add_column("predictions", sa.Column("job_id", sa.Uuid(), nullable=True))
    op.add_column("predictions", sa.Column("position", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_predictions_job_id_jobs", "predictions", "jobs", ["job_id"], ["id"], ondelete="CASCADE"
    )
    # Also serves as the index for reading a job's results in order.
    op.create_unique_constraint(
        "uq_predictions_job_id_position", "predictions", ["job_id", "position"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_predictions_job_id_position", "predictions", type_="unique")
    op.drop_constraint("fk_predictions_job_id_jobs", "predictions", type_="foreignkey")
    op.drop_column("predictions", "position")
    op.drop_column("predictions", "job_id")
    op.drop_table("jobs")
