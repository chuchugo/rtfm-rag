import os
from typing import Any, Callable

from ragapp.config import settings


def _configure_langsmith() -> None:
    """Push LangSmith settings into os.environ so the SDK picks them up.

    pydantic-settings reads the .env file into the Settings object but does NOT
    write undeclared variables into os.environ, so the langsmith SDK would miss
    them without this call.
    """
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)


_configure_langsmith()

try:
    from langsmith import traceable
except ImportError:
    # Graceful no-op if the package is somehow absent at runtime.
    def traceable(**_: Any) -> Callable:  # type: ignore[misc]
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator
