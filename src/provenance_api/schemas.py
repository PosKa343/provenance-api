import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from provenance_api.ml.model import Label


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
