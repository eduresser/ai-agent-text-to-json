"""Public API for text-to-json extraction."""

from __future__ import annotations

from typing import Any, Optional

from text_to_json.agent.graph import create_graph
from text_to_json.agent.state import AgentState
from text_to_json.clients import reset_clients_cache
from text_to_json.settings import get_settings, reset_settings_cache


def _build_initial_state(
    text: str,
    schema: Optional[dict[str, Any]],
    max_iterations_per_chunk: int,
) -> AgentState:
    """Build the initial AgentState for an extraction run."""
    settings = get_settings()
    return {
        "text": text,
        "target_schema": schema,
        "max_iterations": max_iterations_per_chunk,
        "max_chunk_retries": settings.MAX_CHUNK_RETRIES,
        "chunks": [],
        "current_chunk_idx": 0,
        "json_document": {},
        "guidance": {},
        "messages": [],
        "is_chunk_finalized": False,
        "iteration_count": 0,
        "chunk_retry_count": 0,
        "token_usage": {},
    }


def _build_result(final_state: dict[str, Any]) -> dict[str, Any]:
    """Extract the public result dict from the final agent state."""
    return {
        "json_document": final_state.get("json_document", {}),
        "metadata": {
            "total_chunks": len(final_state.get("chunks", [])),
            "final_guidance": final_state.get("guidance", {}),
            "token_usage": final_state.get("token_usage", {}),
        },
        "error": final_state.get("error"),
    }


def extract(
    text: str,
    schema: Optional[dict[str, Any]] = None,
    max_iterations_per_chunk: Optional[int] = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """
    Extract structured data from a text.

    This is the main function of the text-to-json API.

    Args:
        text: The input text for extraction.
        schema: Optional target schema JSON. If not provided, the agent
               will infer a logical structure based on the content.
        max_iterations_per_chunk: Maximum number of iterations of the agent
                                  by chunk before forcing finalization.
                                  If None, uses the value from Settings.MAX_ITERATIONS_PER_CHUNK.
        show_progress: If True, show detailed progress visualization.

    Returns:
        A dictionary with:
        - "json_document": The extracted JSON document.
        - "metadata": Information about the processing:
          - "total_chunks": Total number of chunks processed.
          - "final_guidance": Final guidance used.
          - "token_usage": Token usage information.
        - "error": Error message, if there is one.

    Example:
        >>> from text_to_json import extract
        >>> result = extract(
        ...     text="John Doe, 30 years old, works at Acme Corp...",
        ...     schema={"type": "object", "properties": {"name": {}, "age": {}, "company": {}}}
        ... )
        >>> print(result["json_document"])
        {"name": "John Doe", "age": 30, "company": "Acme Corp"}
    """
    settings = get_settings()

    if max_iterations_per_chunk is None:
        max_iterations_per_chunk = settings.MAX_ITERATIONS_PER_CHUNK

    app = create_graph()
    initial_state = _build_initial_state(text, schema, max_iterations_per_chunk)

    try:
        if show_progress:
            final_state = _run_with_progress(
                app, initial_state, settings.CHAT_MODEL,
                max_iterations_per_chunk, schema,
            )
        else:
            final_state = app.invoke(initial_state)

        return _build_result(final_state)
    finally:
        reset_clients_cache()
        reset_settings_cache()


def _run_with_progress(
    app: Any,
    initial_state: AgentState,
    model_name: str,
    max_iterations_per_chunk: int,
    schema: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Run extraction with Rich live progress display.

    Imports from ``text_to_json.cli`` are deferred so that Rich is only
    required when the caller explicitly requests progress output.
    """
    from text_to_json.cli import (
        print_error_panel,
        print_result_panel,
        print_start_panel,
        run_live_progress,
    )

    text_len = len(initial_state.get("text", ""))
    print_start_panel(model_name, text_len, schema is not None)

    final_state = run_live_progress(
        app, initial_state, model_name, max_iterations_per_chunk,
    )

    token_usage = final_state.get("token_usage", {})
    error = final_state.get("error")

    if error:
        print_error_panel(error)
    else:
        print_result_panel(
            len(final_state.get("chunks", [])),
            len(final_state.get("json_document", {})),
            token_usage,
        )

    return final_state
