from __future__ import annotations
import os


def get_langfuse_callback():
    """Return a Langfuse callback handler, or None if not configured."""
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None
