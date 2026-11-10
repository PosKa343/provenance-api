"""Create predictions table.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("label", sa.String(8), nullable=False),
        sa.Column("probability_ai", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("label IN ('human', 'ai')", name="ck_predictions_label"),
        sa.CheckConstraint(
            "probability_ai >= 0 AND probability_ai <= 1", name="ck_predictions_probability_ai"
        ),
    )


def downgrade() -> None:
    op.drop_table("predictions")
