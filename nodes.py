import os
import time
from typing import Literal
from pydantic import BaseModel


from langchain_chroma import Chroma
from llm import GeminiLLM
from state import GaziMindState
from config import Config


from langchain_core.messages import HumanMessage,RemoveMessage,SystemMessage
from langchain_core.documents import Document




_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")





def _log(msg: str):
    # DEBUG: uvicorn terminaline yazar
    print(f"[DEBUG] {msg}", flush=True)


def _load_propmts(filename:str) ->str:
    with open(os.path.join(_PROMPTS_DIR,filename),"r",encoding="utf-8") as f:
        return f.read()



def _extract_text(content) -> str:

    if isinstance(content,str):
        return content
    if isinstance(content,list):
        return "".join(
         b.get("text","") for b in content if isinstance(b,dict) and b.get("type") == "text"
        )
    return str(content)



def search_documents(vector_store, state:GaziMindState) -> list[Document]:
    retriver = vector_store.as_retriever(search_kwargs={"k":Config.RETRIEVER_K})
    results =  retriver.invoke(state["question"])
    return results


def format_docs(docs) -> str:
    retrives = []

    for d in docs:
        kaynak = d.metadata.get("yonetmelik","Bilinmeyen")
        retrives.append(f"[Kaynak:{kaynak}]\n{d.page_content}")


    return "\n\n --- \n\n".join(retrives)


def _with_summary(system_prompt: str, state: GaziMindState) -> str:
    # Onceki konusmanin ozeti varsa sistem mesajina ekle (summarize node'u eski mesajlari siliyor)
    summary = state.get("summary")
    if not summary:
        return system_prompt
    return f"{system_prompt}\n\n<conversation_summary>\n{summary}\n</conversation_summary>"



class GradeDecision(BaseModel):

    score : Literal["relevant","not_relevant"]
    reasoning: str


class RouteQuery(BaseModel):

    decision : Literal["casual_chat","retrive_docs"]
    reasoning: str


class GaziMindNodes:

    def __init__(self, vector_store : Chroma):
        self.llm = GeminiLLM().get_llm()
        self.vector_store = vector_store
        self.grade_prompt = _load_propmts("grade_doc.txt")
        self.rewrite_prompt = _load_propmts("rewrite_query.txt")
        self.generate_prompt = _load_propmts("generate.txt")
        self.fallback_prompt = _load_propmts("generate_fall_back.txt")
        self.chat_prompt = _load_propmts("gazi_mind.txt")
        self.summarize_prompt = _load_propmts("summarize.txt")
        self.route_prompt = _load_propmts("router.txt")
        self.grader_llm = self.llm.with_structured_output(GradeDecision)
        self.router_llm = self.llm.with_structured_output(RouteQuery)


    def gazi_mind_casual(self,state : GaziMindState) -> dict:

        _log("casual_chat: LLM cagriliyor...")
        t0 = time.time()

        system = SystemMessage(content=_with_summary(self.chat_prompt, state))

        messages = [system] + list(state["messages"])

        response = self.llm.invoke(messages)

        _log(f"casual_chat: cevap geldi ({time.time() - t0:.1f}s)")
        return {"messages":[response]}

    def summarize_node(self,state:GaziMindState):

        existing_summary = state.get("summary","")

        old_messages = state["messages"][0:-2]
        _log(f"summarize: {len(old_messages)} eski mesaj ozetleniyor...")
        t0 = time.time()

        lines = []

        for m in old_messages:
            role = "User" if m.type == "human" else "Assistant"
            lines.append(f"{role}: {_extract_text(m.content)}")


        prompt = self.summarize_prompt.format(
            existing_summary=existing_summary,
            conversation = "\n".join(lines),
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])


        new_summary = _extract_text(response.content)


        delete_ops =  [RemoveMessage(id=m.id) for m in old_messages]
        _log(f"summarize: bitti ({time.time() - t0:.1f}s)")

        return  {"summary":new_summary,"messages":delete_ops}


    def retrive_node(self,state:GaziMindState):

        _log(f"retrive_docs: araniyor -> '{state['question']}'")
        t0 = time.time()

        docs =  search_documents(vector_store=self.vector_store, state=state)

        _log(f"retrive_docs: {len(docs)} dokuman bulundu ({time.time() - t0:.1f}s)")
        for i, d in enumerate(docs):
            _log(f"   [{i}] {d.metadata.get('yonetmelik','?')[:60]} | {d.page_content[:80]!r}")
        return {"documents":docs}

    def grade_documents_node(self,state:GaziMindState):

        if not state.get("documents"):
            _log("grade_docs: dokuman yok -> not_relevant")
            return {"doc_grade":"not_relevant"}

        _log("grade_docs: LLM cagriliyor...")
        t0 = time.time()

        docs_text = format_docs(state["documents"])

        messages = [
            SystemMessage(content=self.grade_prompt),
            HumanMessage(content=f"Question: {state['question']}\n\nDocuments:\n{docs_text}")
        ]

        decision = self.grader_llm.invoke(messages)

        _log(f"grade_docs: {decision.score} ({time.time() - t0:.1f}s) | {decision.reasoning}")
        return {"doc_grade":decision.score}


    def rewrite_query_node(self,state:GaziMindState):

        _log(f"rewrite_query: deneme {state.get('retry_count',0) + 1}, LLM cagriliyor...")
        t0 = time.time()

        messages = [
            SystemMessage(content=self.rewrite_prompt),
            HumanMessage(content=f"Original question: {state['question']}"),
        ]

        response = self.llm.invoke(messages)

        new_question = _extract_text(response.content).strip()
        _log(f"rewrite_query: ({time.time() - t0:.1f}s) yeni soru -> '{new_question}'")

        return{
            "question": new_question,
            "retry_count": state.get("retry_count",0) + 1
        }

    def route_query_node(self,state:GaziMindState):

        _log("router: LLM cagriliyor...")
        t0 = time.time()

        system = SystemMessage(content=_with_summary(self.route_prompt, state))
        messages = [system] + list(state["messages"])
        decision = self.router_llm.invoke(messages)

        _log(f"router: {decision.decision} ({time.time() - t0:.1f}s) | {decision.reasoning}")
        return {"route":decision.decision}

    def generate_node(self, state: GaziMindState) -> dict:
        """Generate a grounded answer using the retrieved policy documents as context."""
        _log("generate: LLM cagriliyor...")
        t0 = time.time()
        docs_text      = format_docs(state["documents"])
        system_content = _with_summary(self.generate_prompt.format(context=docs_text), state)
        messages       = [SystemMessage(content=system_content)] + list(state["messages"])
        response       = self.llm.invoke(messages)
        _log(f"generate: cevap geldi ({time.time() - t0:.1f}s)")
        return {"messages": [response]}


    def fallback_generate_node(self, state: GaziMindState) -> dict:
        """Generate a general answer when no relevant policy documents are found."""
        _log("fallback_generate: LLM cagriliyor...")
        t0 = time.time()
        system_content = _with_summary(self.fallback_prompt, state)
        messages = [SystemMessage(content=system_content)] + list(state["messages"])
        response = self.llm.invoke(messages)
        _log(f"fallback_generate: cevap geldi ({time.time() - t0:.1f}s)")
        return {"messages": [response]}
