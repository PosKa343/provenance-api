import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from provenance_api.ml.model import Label
from provenance_api.models import JobStatus, Plan, Role


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The text to classify.", examples=["The quick brown fox..."])

    @field_validator("text")
    @classmethod
    def text_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class FeatureContributionOut(BaseModel):
    name: str
    value: float = Field(description="The feature's value for this text.")
    contribution: float = Field(
        description="Model weight times value. Positive pushes towards 'ai', negative to 'human'."
    )


class ClassifyResponse(BaseModel):
    prediction_id: uuid.UUID
    label: Label
    probability_ai: float = Field(
        ge=0, le=1, description="Model's probability that the text is AI-generated."
    )
    top_features: list[FeatureContributionOut]
    model_version: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str


# Reused in route decorators so OpenAPI documents the auth failures.
AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorEnvelope, "description": "Missing or invalid credentials."},
    403: {"model": ErrorEnvelope, "description": "Authenticated, but not allowed."},
}


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=10, max_length=128)

    @field_validator("email")
    @classmethod
    def lowercase_email(cls, value: str) -> str:
        return value.lower()


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    role: Role
    plan: Plan
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - a token type, not a password
    expires_in: int = Field(description="Seconds until the token expires.")


class ApiKeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, examples=["laptop"])


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    prefix: str = Field(description="First characters of the key, to tell keys apart.")
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str = Field(description="The full key. Shown only in this response; store it now.")


class UserPage(BaseModel):
    items: list[UserOut]
    next_cursor: str | None = Field(description="Pass as ?cursor= to get the next page.")


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(min_length=1, description="The texts to classify, in order.")

    @field_validator("texts")
    @classmethod
    def texts_have_content(cls, value: list[str]) -> list[str]:
        for position, text in enumerate(value):
            if not text.strip():
                raise ValueError(f"texts[{position}] must contain a non-whitespace character")
        return value


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: JobStatus
    total: int = Field(description="Number of texts submitted.")
    processed: int = Field(description="Number of texts scored so far.")
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    position: int = Field(description="Index of the text in the submitted list.")
    input_sha256: str
    label: Label
    probability_ai: float
    model_version: str


class JobResultsPage(BaseModel):
    items: list[JobResult]
    next_cursor: int | None = Field(description="Pass as ?cursor= to get the next page.")
