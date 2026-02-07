from typing import Any


def update_guidance(
    last_processed_path: str = "",
    current_context: str = "",
    pending_action: str = "",
    extracted_entities_count: int = 0,
) -> dict[str, Any]:
    """
    Finalize the processing of the current chunk and create the guidance for the next.

    This tool MUST be called alone (as the only action) when the current chunk
    has been completely processed and all writes have been confirmed.

    Args:
        last_processed_path: The last JSON path that was processed.
        current_context: Summary of what is being built (e.g. "list of clients").
        pending_action: Pending action for the next chunk (e.g. "expecting_contract_details").
        extracted_entities_count: Number of entities extracted in this chunk.

    Returns:
        The new guidance object to be passed to the next chunk.
    """
    return {
        "finalized": True,
        "guidance": {
            "last_processed_path": last_processed_path,
            "current_context": current_context,
            "pending_action": pending_action,
            "extracted_entities_count": extracted_entities_count,
        },
    }
