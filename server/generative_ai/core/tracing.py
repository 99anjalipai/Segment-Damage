"""
tracing.py

Observability and tracing configuration for the AutoClaim AI pipeline.
Supports LangFuse (open source, self-hostable) and LangSmith (LangChain's platform).

Setup:
    For LangFuse (recommended -- open source):
        pip install langfuse
        Set environment variables:
            LANGFUSE_PUBLIC_KEY=pk-lf-...
            LANGFUSE_SECRET_KEY=sk-lf-...
            LANGFUSE_HOST=https://cloud.langfuse.com  (or your self-hosted URL)

    For LangSmith:
        pip install langsmith
        Set environment variables:
            LANGCHAIN_TRACING_V2=true
            LANGCHAIN_API_KEY=ls-...
            LANGCHAIN_PROJECT=autoclaim-ai

    If neither is configured, tracing is silently disabled.

Usage:
    from generative_ai.core.tracing import get_tracing_callbacks, log_trace_url

    callbacks = get_tracing_callbacks(session_id="claim-123")
    result = graph.invoke(state, config={"callbacks": callbacks})
    log_trace_url()
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger("tracing")


def _is_langfuse_configured() -> bool:
    """Check if LangFuse environment variables are set."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def _is_langsmith_configured() -> bool:
    """Check if LangSmith environment variables are set."""
    return bool(
        os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and os.environ.get("LANGCHAIN_API_KEY")
    )


def get_tracing_callbacks(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: str = "autoclaim-pipeline",
    metadata: Optional[dict] = None,
) -> List:
    """
    Get tracing callback handlers based on available configuration.

    Args:
        session_id: Group multiple traces into a session (e.g., per claim).
        user_id: Associate trace with a user.
        trace_name: Name for the trace in the dashboard.
        metadata: Additional metadata to attach to the trace.

    Returns:
        List of callback handlers. Empty list if no tracing is configured.
    """
    callbacks = []

    # LangFuse
    if _is_langfuse_configured():
        try:
            from langfuse.langchain import CallbackHandler as LangFuseHandler
            import inspect

            # Detect which kwargs the installed version supports
            init_params = inspect.signature(LangFuseHandler.__init__).parameters

            handler_kwargs = {}

            # v2 SDK uses session_id, user_id, trace_name directly
            # v3 SDK uses different parameter names or requires setting via .set_trace_params()
            if "session_id" in init_params:
                if session_id:
                    handler_kwargs["session_id"] = session_id
                if user_id:
                    handler_kwargs["user_id"] = user_id
                if trace_name:
                    handler_kwargs["trace_name"] = trace_name
                if metadata:
                    handler_kwargs["metadata"] = metadata
            
            handler = LangFuseHandler(**handler_kwargs)

            # For v3+, try setting trace params after init
            if "session_id" not in init_params:
                try:
                    if hasattr(handler, "set_trace_params"):
                        params = {}
                        if session_id:
                            params["session_id"] = session_id
                        if user_id:
                            params["user_id"] = user_id
                        if trace_name:
                            params["name"] = trace_name
                        if metadata:
                            params["metadata"] = metadata
                        handler.set_trace_params(**params)
                except Exception:
                    pass  # Tracing still works, just without session grouping

            callbacks.append(handler)
            logger.info(f"[Tracing] LangFuse enabled (session={session_id})")

        except ImportError:
            logger.warning(
                "[Tracing] LANGFUSE env vars set but langfuse package not installed. "
                "Install with: pip install langfuse"
            )
        except Exception as e:
            logger.warning(f"[Tracing] LangFuse init failed: {e}")

    # LangSmith (enabled via env vars automatically, but we can add metadata)
    if _is_langsmith_configured():
        logger.info(
            f"[Tracing] LangSmith enabled (project={os.environ.get('LANGCHAIN_PROJECT', 'default')})"
        )
        # LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true
        # No callback needed, but we log that it's active

    if not callbacks and not _is_langsmith_configured():
        logger.info("[Tracing] No tracing configured. Set LANGFUSE_* or LANGCHAIN_* env vars to enable.")

    return callbacks


def flush_traces():
    """
    Flush any pending traces to the backend.
    Call this after graph execution to ensure all data is sent.
    """
    if _is_langfuse_configured():
        try:
            from langfuse import Langfuse
            client = Langfuse()
            client.flush()
            logger.info("[Tracing] LangFuse traces flushed.")
        except Exception as e:
            logger.warning(f"[Tracing] Flush failed: {e}")


def get_tracing_status() -> dict:
    """
    Return the current tracing configuration status.
    Useful for displaying in the Streamlit UI.
    """
    status = {
        "langfuse_configured": _is_langfuse_configured(),
        "langsmith_configured": _is_langsmith_configured(),
        "active_provider": None,
    }

    if status["langfuse_configured"]:
        status["active_provider"] = "LangFuse"
        status["langfuse_host"] = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    elif status["langsmith_configured"]:
        status["active_provider"] = "LangSmith"
        status["langsmith_project"] = os.environ.get("LANGCHAIN_PROJECT", "default")

    return status