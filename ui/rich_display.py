from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


def create_progress_display(
    current_node: str,
    chunk_idx: int,
    total_chunks: int,
    iteration: int,
    max_iterations: int,
    tools_used: list[str],
    text_preview: str,
    model_name: str,
) -> Table:
    """Build the progress visualization table."""
    node_descriptions = {
        "chunk_text": "Dividing text into chunks...",
        "prepare_chunk": "Preparing chunk for processing...",
        "call_llm": "Calling language model...",
        "execute_tools": "Executing tools...",
        "finalize_chunk": "Finalizing chunk...",
        "__end__": "Processing completed!",
    }

    table = Table(
        title="[bold cyan]Text-to-JSON Agent[/bold cyan]",
        box=box.ROUNDED,
        show_header=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Info", style="bold", width=20)
    table.add_column("Value", style="white")

    status_text = node_descriptions.get(current_node, f"⏳ {current_node}")
    table.add_row("Status", Text(status_text, style="yellow"))

    table.add_row("Model", Text(model_name, style="cyan"))

    if total_chunks > 0:
        chunk_progress = f"[green]{chunk_idx + 1}[/green] / [blue]{total_chunks}[/blue]"
        table.add_row("Chunk", chunk_progress)
    else:
        table.add_row("Chunk", "[dim]Calculating...[/dim]")

    iter_style = "red" if iteration >= max_iterations - 2 else "green"
    table.add_row(
        "Iteration",
        f"[{iter_style}]{iteration}[/{iter_style}] / [blue]{max_iterations}[/blue]",
    )

    if tools_used:
        tools_text = ", ".join(tools_used[-5:])
        if len(tools_used) > 5:
            tools_text = f"...{tools_text}"
        table.add_row("Tools", Text(tools_text, style="magenta"))
    else:
        table.add_row("Tools", "[dim]None yet[/dim]")

    if text_preview:
        preview = text_preview[:60] + "..." if len(text_preview) > 60 else text_preview
        preview = preview.replace("\n", " ")
        table.add_row("Text", Text(preview, style="dim"))

    return table


def print_start_panel(model_name: str, text_len: int, has_schema: bool) -> None:
    """Print the start panel of the extraction."""
    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/bold] {model_name}\n"
            f"[bold]Text:[/bold] {text_len} characters\n"
            f"[bold]Schema:[/bold] {'Provided' if has_schema else 'Automatic inference'}",
            title="[bold cyan]Starting Extraction[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()


def print_result_panel(total_chunks: int, num_fields: int) -> None:
    """Print the success panel at the end of the extraction."""
    console.print(
        Panel(
            f"[bold green]Extraction completed successfully![/bold green]\n\n"
            f"[bold]Chunks processed:[/bold] {total_chunks}\n"
            f"[bold]Fields extracted:[/bold] {num_fields}",
            title="[bold green]Result[/bold green]",
            border_style="green",
        )
    )
    console.print()


def print_error_panel(message: str) -> None:
    """Print the error panel."""
    console.print(
        Panel(
            f"[red]{message}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        )
    )
    console.print()


def print_json_panel(json_document: dict) -> None:
    """Print the extracted JSON in a panel with syntax highlighting."""
    import json

    syntax = Syntax(
        json.dumps(json_document, indent=2, ensure_ascii=False),
        "json",
        theme="monokai",
        line_numbers=True,
    )
    console.print(
        Panel(syntax, title="[bold]Extracted JSON[/bold]", border_style="blue")
    )


def run_live_progress(app, initial_state: dict, model_name: str, max_iterations: int):
    """
    Run the stream of the graph with Live from Rich, updating the progress table.
    Return the final state (with json_document, chunks, guidance, error).
    """
    current_node = "chunk_text"
    total_chunks = 0
    chunk_idx = 0
    iteration = 0
    tools_used: list[str] = []
    current_chunk_text = ""
    final_state = None

    with Live(
        create_progress_display(
            current_node,
            chunk_idx,
            total_chunks,
            iteration,
            max_iterations,
            tools_used,
            current_chunk_text,
            model_name,
        ),
        console=console,
        refresh_per_second=4,
        transient=True,
    ) as live:
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in event.items():
                current_node = node_name

                if "chunks" in state_update and state_update["chunks"]:
                    total_chunks = len(state_update["chunks"])

                if "current_chunk_idx" in state_update:
                    new_chunk_idx = state_update["current_chunk_idx"]
                    if new_chunk_idx != chunk_idx:
                        chunk_idx = new_chunk_idx
                        iteration = 0
                        tools_used = []

                if "current_chunk" in state_update:
                    current_chunk_text = state_update["current_chunk"]

                if "iteration_count" in state_update:
                    iteration = state_update["iteration_count"]

                if "actions_results" in state_update:
                    for result in state_update["actions_results"]:
                        if isinstance(result, dict) and "tool" in result:
                            tool_name = result["tool"]
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)

                if "messages" in state_update:
                    for msg in state_update["messages"]:
                        if hasattr(msg, "tool_calls"):
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get("name", "")
                                if tool_name and tool_name not in tools_used:
                                    tools_used.append(tool_name)

                if "json_document" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["json_document"] = state_update["json_document"]
                if "chunks" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["chunks"] = state_update["chunks"]
                if "guidance" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["guidance"] = state_update["guidance"]
                if "error" in state_update:
                    if final_state is None:
                        final_state = {}
                    final_state["error"] = state_update["error"]

                live.update(
                    create_progress_display(
                        current_node,
                        chunk_idx,
                        total_chunks,
                        iteration,
                        max_iterations,
                        tools_used,
                        current_chunk_text,
                        model_name,
                    )
                )

    return final_state or {}
