import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String, func
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from provenance_api.db import Base


class Role(StrEnum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Plan(StrEnum):
    FREE = "free"
    PRO = "pro"


def _string_enum(enum_class: type[StrEnum], constraint_name: str) -> SqlEnum:
    # Stored as text with a CHECK constraint, not a Postgres enum type, because
    # adding a value to a native enum later needs a more awkward migration.
    return SqlEnum(
        enum_class,
        native_enum=False,
        length=16,
        values_callable=lambda members: [member.value for member in members],
        create_constraint=True,
        name=constraint_name,
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(_string_enum(Role, "ck_users_role"), default=Role.MEMBER)
    plan: Mapped[Plan] = mapped_column(_string_enum(Plan, "ck_users_plan"), default=Plan.FREE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApiKey(Base):
    """A machine credential. Only a hash is stored; the key itself is shown once."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(64))
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PredictionRecord(Base):
    """One classification. Stores a hash of the input, never the text itself."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("label IN ('human', 'ai')", name="ck_predictions_label"),
        CheckConstraint(
            "probability_ai >= 0 AND probability_ai <= 1", name="ck_predictions_probability_ai"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Nullable: rows written before authentication existed have no owner.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    input_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(8))
    probability_ai: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
