"""Structured logging setup. No module in this project should call ``print()``.

Every pipeline run gets a ``run_id`` bound to every subsequent log line, and
logs go to two places at once: a human-readable console stream and a
machine-readable ``logs/{run_id}.jsonl`` file for later inspection.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import cast

import structlog

from config.settings import settings


def new_run_id() -> str:
    """Generate a fresh run identifier."""
    return uuid.uuid4().hex


def configure_logging(run_id: str | None = None, level: int = logging.INFO) -> str:
    """Configure structlog + stdlib logging for console and JSONL file output.

    Idempotent-ish: calling this again replaces the root logger's handlers
    and rebinds the run_id, so a fresh call always fully re-configures
    logging rather than stacking handlers.

    Returns the resolved run_id, now bound to every subsequent log call.
    """
    resolved_run_id = run_id or new_run_id()
    settings.ensure_writable_dirs()
    log_file = settings.logs_dir / f"{resolved_run_id}.jsonl"

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(),
            ],
        )
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers = [console_handler, file_handler]
    root_logger.setLevel(level)

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=resolved_run_id)

    return resolved_run_id


def get_logger(*args: object, **kwargs: object) -> structlog.stdlib.BoundLogger:
    """Thin wrapper over ``structlog.get_logger`` for a consistent import point."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(*args, **kwargs))
