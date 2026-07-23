import uuid
import os
from pathlib import Path


from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate



from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma


from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



EXPORT_TYPE = ExportType.DOC_CHUNKS

llm = ChatOllama(model="llama3.2",temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

files = Path("documents")
chunker = HybridChunker()


chunked_list = []


def docling_load_deneme():
    loader = DoclingLoader(
        file_path="documents/1.pdf",
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID)
    )
    docs = loader.load()
    print(f"Topkam chunk sayısı: {len(docs)}")
    print("="*100)

    for i , doc in enumerate(docs):
        heading = doc.metadata.get("dl_meta",{}).get("headings",["Basliksiz"])
        print(f"\n--- Chunk {i} | Başlık: {heading} ---")
        print(doc.page_content)
        print("-" * 50)


def load_chunks(docs:list):
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="tum_yonetmelikler",
        persist_directory="./chroma_db"
    )
    return vector_store

def search_documents(query:str,vector_store) -> list[Document]:
    retriver = vector_store.as_retriever(search_kwargs={"k":3})

    results =  retriver.invoke(query)
    return results


    

YONETMELIK_ISIMLERI = {
    "1.pdf": "GAZİ ÜNİVERSİTESİ LİSANSÜSTÜ EĞİTİM-ÖĞRETİM VE SINAV YÖNETMELİĞİ",
    "2.pdf": "GAZİ ÜNİVERSİTESİ YABANCI DİL HAZIRLIK EĞİTİMİ YÖNETMELİĞİ",
    "3.pdf": "LİSANS ÖĞRENİMLERİNİ TAMAMLAMAYAN VEYA TAMAMLAYAMAYANLARIN ÖN LİSANS DİPLOMASI ALMALARI VEYA MESLEK YÜKSEKOKULLARINA İNTİBAKLARI HAKKINDA YÖNETMELİK",
    "4.pdf": "MESLEK YÜKSEKOKULLARI VE AÇIKÖĞRETİM ÖN LİSANS PROGRAMLARI MEZUNLARININ LİSANS ÖĞRENİMİNE DEVAMLARI HAKKINDA YÖNETMELİK",
}


def convert_langchain_docs(docpath: Path):

    pdf_files = list(docpath.glob("*.pdf"))
    all_docs = []

    for i in range(len(pdf_files)):
        path = pdf_files[i]
        loader = DoclingLoader(
            file_path=str(path),
            export_type=EXPORT_TYPE,
            chunker=HybridChunker(tokenizer=EMBED_MODEL_ID)
        )
        docs = loader.load()

        dosya_adi = os.path.basename(path)
        # Dict'te varsa gerçek yönetmelik adını kullan, yoksa dosya adını kullan
        yonetmelik_adi = YONETMELIK_ISIMLERI.get(dosya_adi, dosya_adi)
        for doc in docs:
            doc.metadata.pop("dl_meta", None)   # ← EKLENDİ: sorunlu iç içe dict'i sil
            doc.metadata["yonetmelik"] = yonetmelik_adi
            doc.metadata["dosya"] = dosya_adi

        all_docs.extend(docs)
        print(f"{yonetmelik_adi}: {len(docs)} chunk yüklendi.")

    print(f"\nToplam: {len(all_docs)} chunk")
    return all_docs


def indexing():
    all_docs = convert_langchain_docs(files)
    vector_store = load_chunks(all_docs)
    

def open_vector_store():
    return Chroma(
        collection_name="tum_yonetmelikler",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

def format_docs(docs) -> str:
    parcalar = []

    for d in docs:
        kaynak = d.metadata.get("yonetmelik","Bilinmeyen")
        parcalar.append(f"[Kaynak:{kaynak}]\n{d.page_content}")
    

    return "\n\n --- \n\n".join(parcalar)



# [3] PROMPT: halusinasyon guard'li sablon
prompt = ChatPromptTemplate.from_template(
    """Sen bir universite yonetmelik asistanisin.
SADECE asagidaki BAGLAM'a dayanarak cevap ver.
Eger cevap baglamda yoksa "Bu bilgiyi yonetmeliklerde bulamadim." de.
ASLA uydurma, tahmin etme. Cevabin sonunda hangi yonetmelik(ler)e
dayandigini "Kaynak: ..." seklinde belirt.

BAGLAM:
{context}

SORU: {question}

CEVAP:"""
)

def build_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 3})   # ← Runnable, tek girdi alır
    chain = (
        {
            "context": retriever | format_docs,      # soru → ara → formatla
            "question": RunnablePassthrough(),        # soru → aynen geç
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
def main():
    #print("program başladi")
    #print_pdf_files_path(files)
    #docling_load_deneme()

    vs = open_vector_store()
    chain = build_chain(vs)
    cevap = chain.invoke("enver paşa kimdir")
    print(cevap)
   





if __name__ == "__main__":
    main()

