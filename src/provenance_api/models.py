import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
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


class UsageKind(StrEnum):
    CLASSIFY = "classify"
    BATCH = "batch"


class UsageEvent(Base):
    """The durable record of metered usage: one row per charged request."""

    __tablename__ = "usage_events"
    __table_args__ = (Index("ix_usage_events_user_id_occurred_at", "user_id", "occurred_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[UsageKind] = mapped_column(_string_enum(UsageKind, "ck_usage_events_kind"))
    units: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_finished(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED)


class Job(Base):
    """A batch of texts to score in the background. The texts are not stored here."""

    __tablename__ = "jobs"
    __table_args__ = (
        # Makes a retried submission with the same key find the first job
        # instead of creating a second one, even if both arrive at once.
        UniqueConstraint("user_id", "idempotency_key", name="uq_jobs_user_id_idempotency_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        _string_enum(JobStatus, "ck_jobs_status"), default=JobStatus.QUEUED
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    # Hash of the submitted texts, to tell a retry from a reused key.
    request_sha256: Mapped[str] = mapped_column(String(64))
    total: Mapped[int] = mapped_column(Integer)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PredictionRecord(Base):
    """One classification. Stores a hash of the input, never the text itself."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("label IN ('human', 'ai')", name="ck_predictions_label"),
        CheckConstraint(
            "probability_ai >= 0 AND probability_ai <= 1", name="ck_predictions_probability_ai"
        ),
        # One result per text in a job, so re-running part of a job cannot duplicate rows.
        UniqueConstraint("job_id", "position", name="uq_predictions_job_id_position"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Nullable: rows written before authentication existed have no owner.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    # Set only for predictions made by a batch job; position is the text's index in it.
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    position: Mapped[int | None] = mapped_column(Integer)
    input_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(8))
    probability_ai: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
