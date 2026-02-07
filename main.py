import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from settings import get_settings
from ui import console, print_json_panel


def extract(
    text: str,
    schema: Optional[dict[str, Any]] = None,
    max_iterations_per_chunk: int = 20,
    show_progress: bool = False,
) -> dict[str, Any]:
    """
    Extract structured data from a text.

    This is the main function of the agent API.

    Args:
        text: The input text for extraction.
        schema: Optional target schema JSON. If not provided, the agent
               will infer a logical structure based on the content.
        max_iterations_per_chunk: Maximum number of iterations of the agent
                                  by chunk before forcing finalization.
        show_progress: If True, show detailed progress visualization.

    Returns:
        Um dicionário com:
        - "json_document": O documento JSON extraído.
        - "metadata": Information about the processing.
        - "error": Error message, if there is one.

    Example:
        >>> from main import extract
        >>> result = extract(
        ...     text="John Doe, 30 years old, works at Acme Corp...",
        ...     schema={"type": "object", "properties": {"name": {}, "age": {}, "company": {}}}
        ... )
        >>> print(result["json_document"])
        {"name": "John Doe", "age": 30, "company": "Acme Corp"}
    """
    if show_progress:
        return _extract_with_progress(text, schema, max_iterations_per_chunk)
    
    from agent.graph import extract as agent_extract
    return agent_extract(text, schema, max_iterations_per_chunk)


def _extract_with_progress(
    text: str,
    schema: Optional[dict[str, Any]],
    max_iterations_per_chunk: int,
) -> dict[str, Any]:
    """Execute extraction with progress visualization (usa Rich em ui/)."""
    from agent.graph import create_graph
    from agent.state import AgentState
    from ui import (
        print_error_panel,
        print_result_panel,
        print_start_panel,
        run_live_progress,
    )

    settings = get_settings()
    model_name = settings.CHAT_MODEL
    app = create_graph()

    initial_state: AgentState = {
        "text": text,
        "target_schema": schema,
        "max_iterations": max_iterations_per_chunk,
        "chunks": [],
        "current_chunk_idx": 0,
        "json_document": {},
        "guidance": {},
        "messages": [],
        "actions_results": [],
        "is_chunk_finalized": False,
        "iteration_count": 0,
    }

    print_start_panel(model_name, len(text), schema is not None)
    final_state = run_live_progress(
        app, initial_state, model_name, max_iterations_per_chunk
    )

    result = {
        "json_document": final_state.get("json_document", {}),
        "metadata": {
            "total_chunks": len(final_state.get("chunks", [])),
            "final_guidance": final_state.get("guidance", {}),
        },
        "error": final_state.get("error"),
    }

    if result.get("error"):
        print_error_panel(result["error"])
    else:
        print_result_panel(
            result["metadata"]["total_chunks"], len(result["json_document"])
        )

    return result


def main():
    """Entry point of the CLI."""
    parser = argparse.ArgumentParser(
        description="Extract structured data from text in JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from direct text
  text-to-json --text "John Doe, 30 anos, trabalha na Acme Corp"

  # Extract from file
  text-to-json --file documento.txt

  # Use specific schema
  text-to-json --file doc.txt --schema schema.json

  # Save result to file
  text-to-json --file doc.txt --output resultado.json
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t", type=str, help="Direct text for extraction"
    )
    input_group.add_argument(
        "--file", "-f", type=Path, help="Text file for extraction"
    )

    parser.add_argument(
        "--schema",
        "-s",
        type=Path,
        help="Target schema JSON file (optional)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of iterations per chunk (default: 20)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for JSON (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        "-p",
        action="store_true",
        help="Format JSON with indentation",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show detailed progress visualization",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Silent mode (only final result)",
    )

    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = args.file.read_text(encoding="utf-8")

    schema = None
    if args.schema:
        if not args.schema.exists():
            print(f"Error: Schema file not found: {args.schema}", file=sys.stderr)
            sys.exit(1)
        try:
            schema = json.loads(args.schema.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Error: Invalid schema: {e}", file=sys.stderr)
            sys.exit(1)

    show_progress = args.progress and not args.quiet
    try:
        result = extract(
            text=text,
            schema=schema,
            max_iterations_per_chunk=args.max_iterations,
            show_progress=show_progress,
        )
    except Exception as e:
        if not args.quiet:
            console.print(f"[red]Error during extraction: {e}[/red]")
        else:
            print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

    if result.get("error"):
        if not args.quiet:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    indent = 2 if args.pretty else None
    output_json = json.dumps(
        result["json_document"], indent=indent, ensure_ascii=False
    )

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        if not args.quiet:
            if show_progress:
                console.print(f"[green]Result saved in:[/green] {args.output}")
            else:
                print(f"Result saved in: {args.output}")
                print(f"Chunks processed: {result['metadata']['total_chunks']}")
    else:
        if show_progress:
            print_json_panel(result["json_document"])
        else:
            print(output_json)


if __name__ == "__main__":
    main()
