from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


class GeminiLLM:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
        max_output_tokens: int | None = None,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model             = model             or Config.MODEL_NAME,
            temperature       = Config.TEMPERATURE if temperature is None else temperature,
            max_retries       = Config.MAX_RETRIES if max_retries is None else max_retries,
            max_output_tokens = max_output_tokens,
            google_api_key    = Config.API_KEY,
        )

    def get_llm(self):
        return self.llm
