from pydantic import BaseModel
from typing import Optional



class QueryRequest(BaseModel):
    question:str


class QueryResponse(BaseModel):
    answer:str
    sources: Optional[list[str]] = []