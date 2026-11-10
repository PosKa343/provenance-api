import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from provenance_api.db import Base


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
    input_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(8))
    probability_ai: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
