from typing import Any

import jsonpointer


def inspect_keys(document: dict[str, Any], path: str) -> dict[str, Any]:
    """
    Return the keys of an object or the length of an array at a specific path.

    Args:
        document: The root JSON document.
        path: JSON Pointer (RFC 6901) to the location to inspect.

    Returns:
        A dictionary with the information of the node:
        - For objects: {"type": "object", "keys": [...], "count": N}
        - For arrays: {"type": "array", "length": N}
        - For other types: {"type": "...", "value": "..."}
    """
    if not path:
        path = ""

    try:
        if path == "" or path == "/":
            value = document
        else:
            value = jsonpointer.resolve_pointer(document, path)
    except jsonpointer.JsonPointerException as e:
        return {"found": False, "error": str(e), "path": path}

    if isinstance(value, dict):
        keys = list(value.keys())
        return {
            "found": True,
            "path": path,
            "type": "object",
            "keys": keys,
            "count": len(keys),
        }
    elif isinstance(value, list):
        return {
            "found": True,
            "path": path,
            "type": "array",
            "length": len(value),
        }
    else:
        type_name = type(value).__name__
        str_value = str(value)
        if len(str_value) > 100:
            str_value = str_value[:100] + "..."
        return {
            "found": True,
            "path": path,
            "type": type_name,
            "value": str_value,
        }
