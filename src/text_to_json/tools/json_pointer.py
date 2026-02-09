from __future__ import annotations

from urllib.parse import unquote


def decode_pointer_token(token: str) -> str:
    """Decode a single JSON Pointer token (RFC 6901).

    ``~1`` → ``/`` and ``~0`` → ``~``, applied in the correct order.
    """
    return token.replace("~1", "/").replace("~0", "~")


def decode_pointer_token_with_url(token: str) -> str:
    """Decode a JSON Pointer token, also percent-decoding URL-encoded chars."""
    t = decode_pointer_token(token)
    try:
        if "%" in t:
            t = unquote(t)
    except Exception:
        pass
    return t


def encode_pointer_token(token: str) -> str:
    """Encode a string as a JSON Pointer token (RFC 6901).

    ``~`` → ``~0`` and ``/`` → ``~1``, applied in the correct order.
    """
    return str(token).replace("~", "~0").replace("/", "~1")


def parse_json_pointer(path: str) -> list[str]:
    """Parse a JSON Pointer string into a list of decoded tokens.

    Returns an empty list for the root pointers ``""`` and ``"/"``.

    Raises:
        ValueError: If *path* does not start with ``/``.
    """
    if path == "" or path == "/":
        return []
    if not path.startswith("/"):
        raise ValueError(
            f'Invalid JSON Pointer (must start with "/"): {path}'
        )
    return [decode_pointer_token(t) for t in path.split("/")[1:]]


def parse_json_pointer_lenient(path: str) -> list[str]:
    """Like :func:`parse_json_pointer` but auto-prepends ``/`` when missing.

    Never raises on a missing leading slash — instead it silently adds one.
    """
    if not isinstance(path, str):
        raise ValueError("Invalid path: must be a string")
    if path == "" or path == "/":
        return []
    if not path.startswith("/"):
        path = "/" + path
    return [decode_pointer_token(t) for t in path.split("/")[1:]]


def join_pointer(base: str, token: str) -> str:
    """Join a pointer *base* and a *token* into a JSON Pointer string.

    Example::

        >>> join_pointer("", "sections")
        '/sections'
        >>> join_pointer("/sections", "0")
        '/sections/0'
    """
    escaped = encode_pointer_token(token)
    if base == "":
        return "/" + escaped
    return base + "/" + escaped
