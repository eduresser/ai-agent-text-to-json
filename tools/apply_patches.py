from typing import Any

import jsonpatch
import jsonpointer


def apply_patches(
    document: dict[str, Any],
    patches: list[dict[str, Any]],
    target_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply a list of JSON Patch operations to the document.

    Args:
        document: The JSON document to be modified.
        patches: List of JSON Patch operations (RFC 6902).
                 Supported operations: add, remove, replace, move, copy
        target_schema: Optional target schema for validation (not implemented yet).

    Returns:
        A dictionary with the result:
        - Success: {"success": True, "document": {...}, "applied_count": N}
        - Error: {"success": False, "error": "...", "failed_at_index": N}

    Example of patches:
        [
            {"op": "add", "path": "/name", "value": "John"},
            {"op": "replace", "path": "/age", "value": 30},
            {"op": "remove", "path": "/temp_field"},
            {"op": "move", "from": "/old_path", "path": "/new_path"},
            {"op": "copy", "from": "/source", "path": "/destination"}
        ]
    """
    if not patches:
        return {
            "success": True,
            "document": document,
            "applied_count": 0,
            "message": "No patches to apply",
        }

    valid_ops = {"add", "remove", "replace", "move", "copy"}
    for i, patch in enumerate(patches):
        op = patch.get("op")
        if op not in valid_ops:
            return {
                "success": False,
                "error": f"Invalid operation '{op}' at index {i}. Supported: {valid_ops}",
                "failed_at_index": i,
            }
        if op == "test":
            return {
                "success": False,
                "error": "Operation 'test' is not supported. Use inspect_keys/read_value/search_pointer instead.",
                "failed_at_index": i,
            }

    try:
        patch_obj = jsonpatch.JsonPatch(patches)
        new_document = patch_obj.apply(document)

        return {
            "success": True,
            "document": new_document,
            "applied_count": len(patches),
        }
    except jsonpatch.JsonPatchConflict as e:
        return {
            "success": False,
            "error": f"Patch conflict: {e}",
            "failed_at_index": _find_failed_patch_index(document, patches),
        }
    except jsonpatch.JsonPatchException as e:
        return {
            "success": False,
            "error": f"Patch error: {e}",
            "failed_at_index": _find_failed_patch_index(document, patches),
        }
    except jsonpointer.JsonPointerException as e:
        return {
            "success": False,
            "error": f"Invalid path: {e}",
            "failed_at_index": _find_failed_patch_index(document, patches),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "failed_at_index": 0,
        }


def _find_failed_patch_index(
    document: dict[str, Any], patches: list[dict[str, Any]]
) -> int:
    """Try to find which patch failed by applying one by one."""
    current_doc = document
    for i, patch in enumerate(patches):
        try:
            patch_obj = jsonpatch.JsonPatch([patch])
            current_doc = patch_obj.apply(current_doc)
        except Exception:
            return i
    return len(patches) - 1
