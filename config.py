import os

from dotenv import load_dotenv


load_dotenv()




class Config:
    API_KEY           = os.getenv("GEMINI_API_KEY")
    MODEL_NAME        = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
    TEMPERATURE       = float(os.getenv("GEMINI_TEMPERATURE", 0))
    MAX_RETRIES       = int(os.getenv("GEMINI_MAX_RETRIES", 2))
    EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
    RETRIEVER_K       = int(os.getenv("RAG_RETRIEVER_K", 4))
    # Summarise after more than this many messages accumulate in state.
    # At threshold + 2 (the next even count), summarisation fires.
    SUMMARY_THRESHOLD = int(os.getenv("SUMMARY_THRESHOLD", 6))
    DB_PATH           = os.path.join(os.path.dirname(__file__), "gazimind.db")
    MAX_QUERY_RETRIES = int(os.getenv("RAG_MAX_QUERY_RETRIES", 2))
    ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_NAME",MODEL_NAME)
    ROUTER_MAX_TOKENS = os.getenv("ROUTER_MAX_TOKENS",256)