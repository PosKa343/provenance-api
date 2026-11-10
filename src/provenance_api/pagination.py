"""Keyset pagination: the cursor is the sort key of the last row on the page."""

import base64
import binascii
import uuid
from dataclasses import dataclass
from datetime import datetime

from provenance_api.errors import ApiError


@dataclass(frozen=True)
class Cursor:
    created_at: datetime
    id: uuid.UUID

    def encode(self) -> str:
        raw = f"{self.created_at.isoformat()}|{self.id}"
        return base64.urlsafe_b64encode(raw.encode()).decode()

    @classmethod
    def decode(cls, value: str) -> "Cursor":
        try:
            created_at, row_id = base64.urlsafe_b64decode(value.encode()).decode().split("|")
            return cls(created_at=datetime.fromisoformat(created_at), id=uuid.UUID(row_id))
        except (ValueError, binascii.Error, UnicodeDecodeError):
            raise ApiError(400, "invalid_cursor", "The cursor is not valid.") from None
