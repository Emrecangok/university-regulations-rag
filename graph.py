
import sqlite3
import time

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END,START,StateGraph


from langchain_core.messages import HumanMessage

from config import Config
from nodes import GaziMindNodes, _extract_text
from state import GaziMindState
from rag  import open_vector_store
 


def should_summarize(state: GaziMindState) -> str:

    if len(state["messages"]) > Config.SUMMARY_THRESHOLD:
        return "summarize"

    return "end"

def route_by_relevance(state:GaziMindState) -> str:
    if state.get("doc_grade") == "relevant":
        return "generate"
    elif state.get("retry_count",0) >= Config.MAX_QUERY_RETRIES:
        return "fallback_generate" 
    else:
        return "rewrite_query"

def route_desicion(state: GaziMindState) -> str:
    return state.get("route", "retrive_docs")


class GaziMindGraph:

    def __init__(self):
        self.vectorstore = open_vector_store()
        self.nodes = GaziMindNodes(self.vectorstore)
        self.compiled_graph = self._build()


    def _build(self):

        graph = StateGraph(GaziMindState)


        graph.add_node("casual_chat",self.nodes.gazi_mind_casual)
        graph.add_node("router",self.nodes.route_query_node)
        graph.add_node("summarize",self.nodes.summarize_node)

        graph.add_node("retrive_docs",self.nodes.retrive_node)
        graph.add_node("grade_docs",self.nodes.grade_documents_node)
        graph.add_node("rewrite_query",self.nodes.rewrite_query_node)
        graph.add_node("generate",self.nodes.generate_node)
        graph.add_node("fallback_generate",self.nodes.fallback_generate_node)


        graph.add_edge(START,"router")

        graph.add_conditional_edges(
            "router",
            route_desicion,
            {
                "casual_chat":"casual_chat",
                "retrive_docs":"retrive_docs"
            }
        )
        graph.add_conditional_edges(
            "casual_chat",
            should_summarize,
            {
            "summarize":"summarize",
             "end":END
            }
             )
    

        # RETRİVE_DOCS → GRADE DOCS
        graph.add_edge("retrive_docs","grade_docs")
        graph.add_conditional_edges(
            "grade_docs",
            route_by_relevance,
            {
                "generate":"generate",
                "fallback_generate":"fallback_generate",
                "rewrite_query":"rewrite_query"
            }

        )
        graph.add_edge("rewrite_query","retrive_docs")

        graph.add_conditional_edges(
            "generate",
            should_summarize,{
                "summarize":"summarize",
                "end":END
            }
        )
        graph.add_conditional_edges(
                    "fallback_generate",
                    should_summarize,{
                        "summarize":"summarize",
                        "end":END
                    }
                )
        
        graph.add_edge("summarize",END)
        conn = sqlite3.connect(Config.DB_PATH,check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        return graph.compile(checkpointer=checkpointer)


    def get_compiled_graph(self):
        return self.compiled_graph


    def ask(self, question: str, thread_id: str) -> dict:
        print("\n" + "=" * 70, flush=True)
        print(f"[DEBUG] YENI SORU: '{question}' | thread_id={thread_id}", flush=True)
        t0 = time.time()

        # Her turda soruya ozel alanlari sifirla; mesajlar + ozet checkpointer'dan (thread_id) gelir
        try:
            result = self.compiled_graph.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "question": question,
                    "documents": [],
                    "doc_grade": "",
                    "retry_count": 0,
                    "route": "",
                },
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            print(f"[DEBUG] HATA ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:300]}", flush=True)
            print("=" * 70, flush=True)
            raise

        answer = _extract_text(result["messages"][-1].content)
        print(f"[DEBUG] BITTI ({time.time() - t0:.1f}s) | route={result.get('route')} | "
              f"doc_grade={result.get('doc_grade')} | retry={result.get('retry_count')}", flush=True)
        print(f"[DEBUG] CEVAP: {answer[:150]!r}", flush=True)
        print("=" * 70, flush=True)

        sources = []
        if result.get("doc_grade") == "relevant":
            for d in result.get("documents", []):
                kaynak = d.metadata.get("yonetmelik")
                if kaynak and kaynak not in sources:
                    sources.append(kaynak)

        return {"answer": answer, "sources": sources}
        