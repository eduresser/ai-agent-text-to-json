import re
from typing import Any, Literal


def _escape_pointer_token(token: str) -> str:
    """Escape a token for use in JSON Pointer."""
    return token.replace("~", "~0").replace("/", "~1")


def _search_recursive(
    obj: Any,
    current_path: str,
    query: str,
    search_type: Literal["key", "value"],
    fuzzy: bool,
    matches: list[dict[str, Any]],
    limit: int,
    max_value_length: int,
    seen: set[int],
) -> bool:
    """Search recursively in the JSON object."""
    if len(matches) >= limit:
        return True

    obj_id = id(obj)
    if obj_id in seen:
        return False
    if isinstance(obj, (dict, list)):
        seen.add(obj_id)

    if isinstance(obj, dict):
        for key, value in obj.items():
            if len(matches) >= limit:
                return True

            escaped_key = _escape_pointer_token(str(key))
            new_path = f"{current_path}/{escaped_key}"

            if search_type == "key":
                if _matches(str(key), query, fuzzy):
                    value_preview = _preview_value(value, max_value_length)
                    matches.append(
                        {"pointer": new_path, "key": key, "value_preview": value_preview}
                    )

            elif search_type == "value" and not isinstance(value, (dict, list)):
                if _matches(str(value), query, fuzzy):
                    matches.append(
                        {"pointer": new_path, "matched_value": str(value)[:max_value_length]}
                    )

            if isinstance(value, (dict, list)):
                truncated = _search_recursive(
                    value,
                    new_path,
                    query,
                    search_type,
                    fuzzy,
                    matches,
                    limit,
                    max_value_length,
                    seen,
                )
                if truncated:
                    return True

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if len(matches) >= limit:
                return True

            new_path = f"{current_path}/{idx}"

            if search_type == "value" and not isinstance(item, (dict, list)):
                if _matches(str(item), query, fuzzy):
                    matches.append(
                        {"pointer": new_path, "matched_value": str(item)[:max_value_length]}
                    )

            if isinstance(item, (dict, list)):
                truncated = _search_recursive(
                    item,
                    new_path,
                    query,
                    search_type,
                    fuzzy,
                    matches,
                    limit,
                    max_value_length,
                    seen,
                )
                if truncated:
                    return True

    return False


def _matches(text: str, query: str, fuzzy: bool) -> bool:
    """Check if the text matches the query."""
    if not query:
        return True

    text_lower = text.lower()
    query_lower = query.lower()

    if fuzzy:
        query_words = query_lower.split()
        return all(word in text_lower for word in query_words)
    else:
        return query_lower in text_lower


def _preview_value(value: Any, max_length: int) -> str:
    """Create a preview of the value for the result."""
    if isinstance(value, dict):
        return f"{{...}} ({len(value)} keys)"
    elif isinstance(value, list):
        return f"[...] ({len(value)} items)"
    else:
        str_val = str(value)
        if len(str_val) > max_length:
            return str_val[:max_length] + "..."
        return str_val


def search_pointer(
    document: dict[str, Any],
    query: str,
    search_type: Literal["key", "value"] = "value",
    fuzzy_match: bool = False,
    limit: int = 20,
    max_value_length: int = 120,
) -> dict[str, Any]:
    """
    Search for keys or values in the JSON document and return JSON Pointers.

    Args:
        document: The root JSON document.
        query: Search query.
        search_type: "key" to search in keys, "value" to search in values.
        fuzzy_match: If True, fuzzy search (all words present).
        limit: Maximum number of results.
        max_value_length: Maximum length of the preview of values.

    Returns:
        A dictionary with the results:
        {"matches": [...], "count": N, "truncated": bool}
    """
    matches: list[dict[str, Any]] = []
    seen: set[int] = set()

    truncated = _search_recursive(
        document,
        "",
        query,
        search_type,
        fuzzy_match,
        matches,
        limit,
        max_value_length,
        seen,
    )

    return {
        "query": query,
        "type": search_type,
        "fuzzy": fuzzy_match,
        "matches": matches,
        "count": len(matches),
        "truncated": truncated,
    }
