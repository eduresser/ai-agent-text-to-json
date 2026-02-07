from typing import Any

import jsonpointer


def _truncate_value(
    value: Any,
    max_string_length: int = 160,
    max_depth: int = 6,
    max_array_items: int = 50,
    max_object_keys: int = 50,
    current_depth: int = 0,
) -> Any:
    """Truncate values to avoid too large responses."""
    if current_depth >= max_depth:
        if isinstance(value, dict):
            return f"{{...}} ({len(value)} keys)"
        elif isinstance(value, list):
            return f"[...] ({len(value)} items)"
        elif isinstance(value, str) and len(value) > max_string_length:
            return value[:max_string_length] + "..."
        return value

    if isinstance(value, str):
        if len(value) > max_string_length:
            return value[:max_string_length] + f"... ({len(value)} chars total)"
        return value

    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) > max_object_keys:
            truncated = {
                k: _truncate_value(
                    value[k],
                    max_string_length,
                    max_depth,
                    max_array_items,
                    max_object_keys,
                    current_depth + 1,
                )
                for k in keys[:max_object_keys]
            }
            truncated["__truncated__"] = f"{len(keys) - max_object_keys} more keys"
            return truncated
        return {
            k: _truncate_value(
                v,
                max_string_length,
                max_depth,
                max_array_items,
                max_object_keys,
                current_depth + 1,
            )
            for k, v in value.items()
        }

    if isinstance(value, list):
        if len(value) > max_array_items:
            truncated = [
                _truncate_value(
                    item,
                    max_string_length,
                    max_depth,
                    max_array_items,
                    max_object_keys,
                    current_depth + 1,
                )
                for item in value[:max_array_items]
            ]
            truncated.append(f"... ({len(value) - max_array_items} more items)")
            return truncated
        return [
            _truncate_value(
                item,
                max_string_length,
                max_depth,
                max_array_items,
                max_object_keys,
                current_depth + 1,
            )
            for item in value
        ]

    return value


def read_value(
    document: dict[str, Any],
    path: str,
    max_string_length: int = 160,
    max_depth: int = 6,
    max_array_items: int = 50,
    max_object_keys: int = 50,
) -> dict[str, Any]:
    """
    Read the value at a specific path in the JSON document.

    Args:
        document: The root JSON document.
        path: JSON Pointer (RFC 6901) to the value.
        max_string_length: Maximum length of strings before truncating.
        max_depth: Maximum depth for nested objects.
        max_array_items: Maximum number of array items to return.
        max_object_keys: Maximum number of object keys to return.

    Returns:
        A dictionary with the result:
        - Success: {"found": True, "path": "...", "value": ...}
        - Error: {"found": False, "error": "...", "path": "..."}
    """

    try:
        if path == "" or path == "/":
            value = document
        else:
            value = jsonpointer.resolve_pointer(document, path)
    except jsonpointer.JsonPointerException as e:
        return {"found": False, "error": str(e), "path": path}

    truncated_value = _truncate_value(
        value, max_string_length, max_depth, max_array_items, max_object_keys
    )

    return {
        "found": True,
        "path": path,
        "value": truncated_value,
        "type": type(value).__name__,
    }