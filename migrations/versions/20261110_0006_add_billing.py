"""Add Stripe customer link to users and a table of handled Stripe events.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("stripe_customer_id", sa.String(64), nullable=True))
    op.add_column("users", sa.Column("billing_event_at", sa.DateTime(timezone=True), nullable=True))
    op.create_unique_constraint("uq_users_stripe_customer_id", "users", ["stripe_customer_id"])

    op.create_table(
        "stripe_events",
        # Stripe's own event id. The primary key is what makes handling idempotent.
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("type", sa.String(100), nullable=False),
        sa.Column(
            "processed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("stripe_events")
    op.drop_constraint("uq_users_stripe_customer_id", "users", type_="unique")
    op.drop_column("users", "billing_event_at")
    op.drop_column("users", "stripe_customer_id")
