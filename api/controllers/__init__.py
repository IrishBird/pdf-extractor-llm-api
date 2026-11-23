import os

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage

from api.graphs.chat_graph import ChatGraph
from api.models.states import ChatState
from api.stream.orchestrator import StreamOrchestrator
from api.models.schema import ChatRequest
from api.repository.repo_provider import get_graph

router = APIRouter(prefix="/v1", tags=["assistant"])


@router.post("/assistant")
async def get_assistant_data(
    request: ChatRequest,
    chat_graph: ChatGraph = Depends(get_graph),
):

    graph = chat_graph.graph.build_graph().compile()
    orchestrator = StreamOrchestrator()
    state = ChatState(
        user_input=request.user_prompt,
        messages=[HumanMessage(content=request.user_prompt)],
    )
    result = await orchestrator.accumulate(graph, state)
    return result
