import os
from typing import Any


class _NoOpLangfuse:
    def flush(self) -> None:
        return None


class _NoOpCallbackHandler:
    pass


def _noop_observe(*_args: Any, **_kwargs: Any):
    def decorator(func):
        return func

    return decorator


try:
    from langfuse import Langfuse, observe
    from langfuse.langchain import CallbackHandler
except ImportError:
    Langfuse = None
    CallbackHandler = _NoOpCallbackHandler
    observe = _noop_observe


_langfuse_instance = None


def _build_langfuse_client():
    if Langfuse is None:
        return _NoOpLangfuse()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return _NoOpLangfuse()

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )


def get_langfuse_client():
    global _langfuse_instance
    if _langfuse_instance is None:
        _langfuse_instance = _build_langfuse_client()
    return _langfuse_instance


class _LangfuseProxy:
    def flush(self) -> None:
        get_langfuse_client().flush()


langfuse_client = _LangfuseProxy()


def build_callback_handler():
    client = get_langfuse_client()
    if isinstance(client, _NoOpLangfuse):
        return _NoOpCallbackHandler()

    try:
        return CallbackHandler(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"))
    except Exception:
        return _NoOpCallbackHandler()


__all__ = ["langfuse_client", "get_langfuse_client", "build_callback_handler", "observe"]
