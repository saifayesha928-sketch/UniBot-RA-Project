from __future__ import annotations

import logging

import structlog


def configure_logging(*, file: object | None = None) -> None:
    import sys
    from typing import cast
    from io import TextIOWrapper

    stream = file if file is not None else sys.stdout
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=cast(TextIOWrapper, stream), force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIOWrapper, stream)),
        cache_logger_on_first_use=False,
    )
