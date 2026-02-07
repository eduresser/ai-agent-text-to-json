"""Definição do grafo LangGraph para extração de dados estruturados."""

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from agent.nodes import (
    call_llm_node,
    chunk_text_node,
    execute_tools_node,
    finalize_chunk_node,
    has_more_chunks,
    is_chunk_done,
    prepare_chunk_node,
)
from agent.state import AgentState


def create_graph() -> StateGraph:
    """
    Cria o grafo LangGraph para extração de dados estruturados.

    Returns:
        O grafo compilado pronto para execução.
    """
    # Criar o grafo de estado
    graph = StateGraph(AgentState)

    # Adicionar nós
    graph.add_node("chunk_text", chunk_text_node)
    graph.add_node("prepare_chunk", prepare_chunk_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("finalize_chunk", finalize_chunk_node)

    # Definir o ponto de entrada
    graph.set_entry_point("chunk_text")

    # Adicionar edges
    graph.add_edge("chunk_text", "prepare_chunk")

    # Após preparar, verificar se há chunks
    graph.add_conditional_edges(
        "prepare_chunk",
        has_more_chunks,
        {
            "call_llm": "call_llm",
            "__end__": END,
        },
    )

    # Após chamar LLM, executar ferramentas
    graph.add_edge("call_llm", "execute_tools")

    # Após executar ferramentas, verificar se chunk está pronto
    graph.add_conditional_edges(
        "execute_tools",
        is_chunk_done,
        {
            "finalize_chunk": "finalize_chunk",
            "call_llm": "call_llm",
        },
    )

    # Após finalizar chunk, voltar para preparar o próximo
    graph.add_edge("finalize_chunk", "prepare_chunk")

    return graph.compile()


def extract(
    text: str,
    schema: Optional[dict[str, Any]] = None,
    max_iterations_per_chunk: int = 20,
) -> dict[str, Any]:
    """
    Extrai dados estruturados de um texto.

    Esta é a função principal da API do agente.

    Args:
        text: O texto de entrada para extração.
        schema: Schema JSON alvo opcional. Se não fornecido, o agente
               inferirá uma estrutura lógica baseada no conteúdo.
        max_iterations_per_chunk: Número máximo de iterações do agente
                                  por chunk antes de forçar finalização.

    Returns:
        Um dicionário com:
        - "json_document": O documento JSON extraído.
        - "metadata": Informações sobre o processamento.
        - "error": Mensagem de erro, se houver.

    Exemplo:
        >>> from agent import extract
        >>> result = extract(
        ...     text="John Doe, 30 years old, works at Acme Corp...",
        ...     schema={"type": "object", "properties": {"name": {}, "age": {}, "company": {}}}
        ... )
        >>> print(result["json_document"])
        {"name": "John Doe", "age": 30, "company": "Acme Corp"}
    """
    # Criar o grafo
    app = create_graph()

    # Preparar estado inicial
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

    # Executar o grafo
    final_state = app.invoke(initial_state)

    # Preparar resultado
    return {
        "json_document": final_state.get("json_document", {}),
        "metadata": {
            "total_chunks": len(final_state.get("chunks", [])),
            "final_guidance": final_state.get("guidance", {}),
        },
        "error": final_state.get("error"),
    }
