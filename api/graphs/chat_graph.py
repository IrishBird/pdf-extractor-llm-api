from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from api.components.assistants import MultiModalAsssitant
from api.components.message_strategies import SingleImageStrategy
from api.components.tools import ClaimBaseTool
from api.graphs.base_graph import BaseGraph
from api.models.states import ChatState


@dataclass
class PreLLMState:
    user_input: str
    allow: bool = False
    reason: Optional[str] = None


def _read(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _includes_tool_call(message: Any) -> bool:
    if getattr(message, "tool_calls", None):
        return True
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") in {"tool_call", "tool_use"}
        for block in content
    )


class ChatGraph(BaseGraph[ChatState]):
    def __init__(
        self,
        provider: str,
        model: str,
        tools: Optional[List[BaseTool]] = None,
        message_strategy: Any = None,
    ):
        self.provider = provider
        self.model = model
        self.tools = tools if tools is not None else ClaimBaseTool().load_tools()
        self.message_strategy = message_strategy or SingleImageStrategy()
        self.assistant = MultiModalAsssitant(
            provider=provider,
            model=model,
            tools=self.tools,
            message_strategy=self.message_strategy,
        )

    def should_continue(self, state: ChatState) -> str:
        messages = _read(state, "messages") or []
        if not messages:
            return END
        last_message = messages[-1]
        if _includes_tool_call(last_message):
            return "tools"
        return END

    def execute_tool(self, state: ChatState) -> dict:
        messages = list(_read(state, "messages") or [])
        tool_node = ToolNode(self.tools or [])
        result = tool_node.invoke({"messages": messages})
        if isinstance(result, dict) and "messages" in result:
            return {"messages": messages + list(result["messages"])}
        return result

    def pre_llm_continue(self, state: ChatState) -> PreLLMState:
        user_input = _read(state, "user_input") or ""
        allowed = bool(user_input.strip() or _read(state, "messages"))
        return PreLLMState(
            user_input=user_input,
            allow=allowed,
            reason=None if allowed else "Request is not allowed without user input.",
        )

    def pre_llm(self, state: ChatState) -> dict:
        decision = self.pre_llm_continue(state)
        if decision.allow:
            return {"is_error": False, "error": None}
        return {
            "is_error": True,
            "is_success": False,
            "error": decision.reason,
        }

    def route_pre_llm(self, state: ChatState) -> str:
        if _read(state, "is_error"):
            return END
        return "assistant"

    def build_graph(self) -> StateGraph[ChatState]:
        graph = StateGraph(state_schema=ChatState)
        graph.add_node("pre_llm", self.pre_llm)
        graph.add_node("assistant", self.assistant)
        graph.add_node("tools", self.execute_tool)

        graph.add_edge(START, "pre_llm")
        graph.add_conditional_edges(
            "pre_llm",
            self.route_pre_llm,
            {"assistant": "assistant", END: END},
        )
        graph.add_conditional_edges(
            "assistant",
            self.should_continue,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "assistant")

        return graph
