# Stub — populated when Treatment 7 (Mem0 LTM) is implemented.

from langchain_core.tools import tool


@tool
def query_long_term_memory(query: str) -> str:
    """Query structured long-term memory for relevant prior learnings. (Treatment 7 only)"""
    return "Long-term memory not yet initialised."
