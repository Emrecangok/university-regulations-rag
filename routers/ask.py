from fastapi import APIRouter
from models import QueryResponse,QueryRequest
from rag import build_chain, open_vector_store



router = APIRouter(prefix="/task",tags=["Ask"])


vs = open_vector_store()
chain = build_chain(vs)


@router.post("/",response_model=QueryResponse)
def ask(request: QueryRequest):
    answer = chain.invoke(request.question)
    return {"answer":answer}