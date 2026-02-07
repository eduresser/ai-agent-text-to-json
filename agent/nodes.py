import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import (
    build_observation_message,
    build_system_prompt,
    build_user_message,
)
from agent.state import AgentState
from clients import get_json_chat_model
from chunking.semantic import chunk_with_fallback
from tools.apply_patches import apply_patches
from tools.inspect_keys import inspect_keys
from tools.read_value import read_value
from tools.search_pointer import search_pointer
from tools.update_guidance import update_guidance


def chunk_text_node(state: AgentState) -> dict[str, Any]:
    """
    Node that divides the text into semantic chunks.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state with the chunks.
    """
    text = state.get("text", "")

    if not text:
        return {
            "chunks": [],
            "current_chunk_idx": 0,
            "json_document": {},
            "error": "No text provided for processing.",
        }

    chunks = chunk_with_fallback(text)

    return {
        "chunks": chunks,
        "current_chunk_idx": 0,
        "json_document": {},
        "guidance": {},
        "is_chunk_finalized": False,
        "iteration_count": 0,
        "max_iterations": state.get("max_iterations", 20),
    }


def prepare_chunk_node(state: AgentState) -> dict[str, Any]:
    """
    Node that prepares the next chunk for processing.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state for the new chunk.
    """
    chunks = state.get("chunks", [])
    current_idx = state.get("current_chunk_idx", 0)

    if current_idx >= len(chunks):
        return {"current_chunk": ""}

    current_chunk = chunks[current_idx]

    system_prompt = build_system_prompt(
        target_schema=state.get("target_schema"),
        previous_guidance=state.get("guidance"),
        json_skeleton=state.get("json_document", {}),
    )

    user_message = build_user_message(
        text_chunk=current_chunk,
        chunk_index=current_idx,
        total_chunks=len(chunks),
    )

    return {
        "current_chunk": current_chunk,
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
        "actions_results": [],
        "is_chunk_finalized": False,
        "iteration_count": 0,
    }


def call_llm_node(state: AgentState) -> dict[str, Any]:
    """
    Node that calls the LLM to process the current chunk.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state with the LLM response.
    """
    messages = state.get("messages", [])

    actions_results = state.get("actions_results", [])
    if actions_results:
        observation = build_observation_message(actions_results)
        messages = list(messages) + [HumanMessage(content=observation)]

    llm = get_json_chat_model()
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """
    Node that executes the tools requested by the LLM.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state with the results of the tools.
    """
    messages = state.get("messages", [])

    last_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_message = msg
            break

    if not last_message:
        return {"error": "No LLM response found."}

    try:
        response_data = json.loads(last_message.content)
    except json.JSONDecodeError as e:
        return {"error": f"Error parsing LLM response: {e}"}

    think = response_data.get("think", "")
    actions = response_data.get("actions", [])

    if not actions:
        return {"error": "No actions specified by the LLM."}

    if len(actions) == 1 and actions[0].get("action") == "update_guidance":
        guidance_input = actions[0].get("input", {})
        guidance_result = update_guidance(**guidance_input)

        return {
            "guidance": guidance_result["guidance"],
            "is_chunk_finalized": True,
            "actions_results": [{"action": "update_guidance", "result": guidance_result}],
        }

    document = state.get("json_document", {})
    results = []
    new_document = document

    for action_item in actions:
        action_name = action_item.get("action")
        action_input = action_item.get("input", {})

        if action_name == "inspect_keys":
            result = inspect_keys(new_document, action_input.get("path", ""))
            results.append({"action": action_name, "input": action_input, "result": result})

        elif action_name == "read_value":
            result = read_value(
                new_document,
                action_input.get("path", ""),
                action_input.get("max_string_length", 160),
                action_input.get("max_depth", 6),
                action_input.get("max_array_items", 50),
                action_input.get("max_object_keys", 50),
            )
            results.append({"action": action_name, "input": action_input, "result": result})

        elif action_name == "search_pointer":
            result = search_pointer(
                new_document,
                action_input.get("query", ""),
                action_input.get("type", "value"),
                action_input.get("fuzzy_match", False),
                action_input.get("limit", 20),
                action_input.get("max_value_length", 120),
            )
            results.append({"action": action_name, "input": action_input, "result": result})

        elif action_name == "apply_patches":
            patches = action_input.get("patches", [])
            result = apply_patches(
                new_document,
                patches,
                state.get("target_schema"),
            )
            if result["success"]:
                new_document = result["document"]
            results.append({"action": action_name, "input": action_input, "result": result})

        elif action_name == "update_guidance":
            results.append({
                "action": action_name,
                "input": action_input,
                "result": {
                    "error": "update_guidance must be the ONLY action when finalizing. "
                    "Other actions were present in the same response."
                },
            })

        else:
            results.append({
                "action": action_name,
                "input": action_input,
                "result": {"error": f"Unknown action: {action_name}"},
            })

    return {
        "json_document": new_document,
        "actions_results": results,
    }


def finalize_chunk_node(state: AgentState) -> dict[str, Any]:
    """
    Node that finalizes the current chunk processing and advances to the next.

    Args:
        state: Current state of the agent.

    Returns:
        Updates to the state for the next chunk.
    """
    current_idx = state.get("current_chunk_idx", 0)

    return {
        "current_chunk_idx": current_idx + 1,
        "is_chunk_finalized": False,
        "actions_results": [],
    }


def has_more_chunks(state: AgentState) -> Literal["call_llm", "__end__"]:
    """
    Check if there are more chunks to process.

    Args:
        state: Current state of the agent.

    Returns:
        "call_llm" if there are more chunks, "__end__" if not.
    """
    chunks = state.get("chunks", [])
    current_idx = state.get("current_chunk_idx", 0)

    if state.get("error"):
        return "__end__"

    if current_idx < len(chunks):
        return "call_llm"

    return "__end__"


def is_chunk_done(state: AgentState) -> Literal["finalize_chunk", "call_llm"]:
    """
    Check if the current chunk has been finalized or needs more iterations.

    Args:
        state: Current state of the agent.

    Returns:
        "finalize_chunk" if finalized, "call_llm" if needs more iterations.
    """
    if state.get("is_chunk_finalized"):
        return "finalize_chunk"

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 20)

    if iteration_count >= max_iterations:
        return "finalize_chunk"

    if state.get("error"):
        return "finalize_chunk"

    return "call_llm"
