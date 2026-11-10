"""Link predictions to the user who requested them.

The column is nullable because predictions made before authentication
existed have no owner.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_predictions_user_id_users",
        "predictions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_predictions_user_id", "predictions", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_user_id", table_name="predictions")
    op.drop_constraint("fk_predictions_user_id_users", "predictions", type_="foreignkey")
    op.drop_column("predictions", "user_id")
