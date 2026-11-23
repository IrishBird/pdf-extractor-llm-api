from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from langgraph.graph import StateGraph
from pydantic import BaseModel

S = TypeVar("S", bound=BaseModel)


class BaseGraph(ABC, Generic[S]):
    def __init__(self, state_schema: Type[S]):
        self.state_schema = state_schema
        self.graph = StateGraph(state_schema)

    @abstractmethod
    def build_graph(self) -> StateGraph[S]:
        pass
