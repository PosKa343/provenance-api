"""Structured logging: each line is one JSON object tagged with its request or job."""

import logging
import sys
from typing import Literal

import structlog
from structlog.typing import Processor

_HANDLER_NAME = "provenance-structured"

# Applied to every record, whether it comes from structlog or from the
# standard library (uvicorn, arq, SQLAlchemy and this package's own loggers).
# merge_contextvars adds whatever is bound for the current request or job.
_SHARED: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def formatter(log_format: Literal["json", "console"] = "json") -> logging.Formatter:
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )


def configure_logging(log_format: Literal["json", "console"] = "json", level: str = "INFO") -> None:
    """Route all logging through one stdout handler. Safe to call more than once."""
    structlog.configure(
        processors=[*_SHARED, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(formatter(log_format))

    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # These libraries install their own plain-text handlers; send them to ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "arq.worker", "arq.jobs"):
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True
