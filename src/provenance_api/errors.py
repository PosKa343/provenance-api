"""Every error leaves the API in one shape: {"error": {"code", "message", "request_id"}}."""

import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from provenance_api.schemas import ErrorBody, ErrorEnvelope

REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger(__name__)

_CODES_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
}


class ApiError(Exception):
    """Raise from a route to return a specific error code and status."""

    def __init__(
        self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers or {}


def unauthorized(message: str, scheme: str) -> ApiError:
    return ApiError(401, "unauthorized", message, headers={"WWW-Authenticate": scheme})


def forbidden(message: str) -> ApiError:
    return ApiError(403, "forbidden", message)


def not_found(message: str) -> ApiError:
    return ApiError(404, "not_found", message)


def _request_id(request: Request) -> str:
    return str(request.state.request_id)


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, request_id=request_id))
    return JSONResponse(
        envelope.model_dump(),
        status_code=status_code,
        headers={**(headers or {}), REQUEST_ID_HEADER: request_id},
    )


async def _attach_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = _request_id(request)
    return response


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)  # noqa: S101 - narrows the type for mypy
    return _error_response(request, exc.status_code, exc.code, exc.message, exc.headers)


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    code = _CODES_BY_STATUS.get(exc.status_code, "http_error")
    return _error_response(request, exc.status_code, code, str(exc.detail))


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    return _error_response(request, 422, "validation_error", f"{field}: {first['msg']}")


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", extra={"request_id": _request_id(request)})
    return _error_response(request, 500, "internal_error", "Internal server error.")


def install_error_handling(app: FastAPI) -> None:
    app.middleware("http")(_attach_request_id)
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
