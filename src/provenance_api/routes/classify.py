import hashlib

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from provenance_api.dependencies import (
    ApiKeyUser,
    ClassifierDep,
    LimiterDep,
    SessionDep,
    SettingsDep,
)
from provenance_api.errors import ApiError
from provenance_api.metrics import PREDICTIONS
from provenance_api.models import PredictionRecord, UsageEvent, UsageKind
from provenance_api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorEnvelope,
    FeatureContributionOut,
)

router = APIRouter(prefix="/v1", tags=["classify"])


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify one text as human-written or AI-generated",
    responses={
        401: {"model": ErrorEnvelope},
        413: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        429: {"model": ErrorEnvelope, "description": "Rate limit or daily quota reached."},
    },
)
async def classify(
    body: ClassifyRequest,
    user: ApiKeyUser,
    classifier: ClassifierDep,
    session: SessionDep,
    settings: SettingsDep,
    limiter: LimiterDep,
) -> ClassifyResponse:
    if len(body.text) > settings.max_text_chars:
        raise ApiError(
            413, "text_too_long", f"text must be at most {settings.max_text_chars} characters."
        )

    # Charged here, after validation, so a rejected request costs no quota.
    await limiter.consume_quota(user.id, 1, settings.limits_for(user.plan))

    # Feature extraction and scoring are CPU-bound. Running them here would
    # block the event loop for every other request, so they go to a worker thread.
    prediction = await run_in_threadpool(classifier.predict, body.text)

    record = PredictionRecord(
        user_id=user.id,
        input_sha256=hashlib.sha256(body.text.encode("utf-8")).hexdigest(),
        label=prediction.label.value,
        probability_ai=prediction.probability_ai,
        model_version=prediction.model_version,
    )
    session.add(record)
    session.add(UsageEvent(user_id=user.id, kind=UsageKind.CLASSIFY, units=1))
    await session.commit()
    PREDICTIONS.labels(prediction.label.value, "single").inc()

    return ClassifyResponse(
        prediction_id=record.id,
        label=prediction.label,
        probability_ai=prediction.probability_ai,
        top_features=[
            FeatureContributionOut(name=f.name, value=f.value, contribution=f.contribution)
            for f in prediction.top_features
        ],
        model_version=prediction.model_version,
    )
