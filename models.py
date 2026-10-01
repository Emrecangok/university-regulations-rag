from pydantic import BaseModel
from typing import Optional



class QueryRequest(BaseModel):
    question:str
    thread_id: Optional[str] = None   # ayni sohbetin mesajlari bu id ile hatirlanir


class QueryResponse(BaseModel):
    answer:str
    sources: Optional[list[str]] = []