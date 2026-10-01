import uuid

from fastapi import APIRouter
from models import QueryResponse,QueryRequest
from graph import GaziMindGraph



router = APIRouter(prefix="/task",tags=["Ask"])


gazi_mind = GaziMindGraph()


@router.post("/",response_model=QueryResponse)
def ask(request: QueryRequest):
    # thread_id yoksa tek seferlik (hafizasiz) sohbet
    thread_id = request.thread_id or str(uuid.uuid4())
    return gazi_mind.ask(request.question, thread_id)
