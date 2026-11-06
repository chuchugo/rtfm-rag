from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://ragapp:ragapp@localhost:5432/ragapp"

    # Which LLMProvider create_llm_client() builds: "ollama" (local),
    # "openai" (OpenAI or any OpenAI-compatible endpoint), or "anthropic".
    llm_provider: str = "openai"

    # Ollama (only needed when llm_provider="ollama")
    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.2:3b"
    ollama_keep_alive: str = "30m"
    ollama_chat_num_ctx: int = 2048

    # OpenAI-compatible provider (chat + embeddings). base_url lets this point
    # at any OpenAI-compatible endpoint, not just OpenAI itself.
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # FAISS vector index directory (chunks stored here, not in Postgres)
    faiss_index_path: str = "./data/faiss"

    # Anthropic provider (used when llm_provider="anthropic").
    anthropic_api_key: str | None = None
    anthropic_chat_model: str = "claude-haiku-4-5-20251001"

    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieval_top_k: int = 5

    # Conversation memory: number of Q&A pairs to keep verbatim before summarizing
    max_history_turns: int = 10

    # LangSmith tracing (optional — tracing is off when langsmith_api_key is unset)
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "rtfm-rag"


settings = Settings()
