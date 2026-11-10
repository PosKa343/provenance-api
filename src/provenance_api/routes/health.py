from fastapi import APIRouter

from provenance_api.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness check")
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")
