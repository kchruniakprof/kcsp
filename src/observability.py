"""
Centralny logger dla całego pipeline.
Każdy krok loguje JSON na stderr — nie miesza z promptfoo stdout.
"""
from __future__ import annotations

import sys
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
