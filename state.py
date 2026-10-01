from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GaziMindState(TypedDict):

    messages : Annotated[list[BaseMessage],add_messages]

    context : str

    question: str

    summary : str

    documents : list[Document]

    doc_grade: str

    retry_count: int

    route: str


