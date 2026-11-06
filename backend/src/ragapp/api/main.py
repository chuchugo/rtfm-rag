import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ragapp.chat.clarifier import check_needs_clarification
from ragapp.config import settings
from ragapp.eval.heuristics import extract_citations, is_refusal
from ragapp.llm.base import SUMMARY_SYSTEM_PROMPT, Message
from ragapp.llm.factory import create_llm_client
from ragapp.retrieval.retriever import Retriever
from ragapp.retrieval.store import VectorStore, init_schema

app = FastAPI(title="rtfm-rag")
init_schema()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    # Also allow any 192.168.x.x:5173 origin so devices on the same home LAN
    # (e.g. a phone) can hit the dev server. This is a local-only convenience,
    # not meant to survive past dev — tighten before this is ever exposed
    # beyond a trusted home network.
    allow_origin_regex=r"http://192\.168\.\d{1,3}\.\d{1,3}:5173",
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
llm_client = create_llm_client()
store = VectorStore()


def _apply_history_window(history: list[Message], llm_client) -> list[Message]:
    """Keep at most max_history_turns Q&A pairs verbatim.
    When exceeded, summarize the overflow with the LLM and inject the summary
    as a leading message pair so the model retains product/question context."""
    max_messages = settings.max_history_turns * 2
    if len(history) <= max_messages:
        return history

    overflow = history[: len(history) - max_messages]
    recent = history[len(history) - max_messages :]

    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in overflow)
    summary = "".join(
        llm_client.chat_stream(
            user_message=transcript,
            context="",
            system_prompt=SUMMARY_SYSTEM_PROMPT,
        )
    )

    return [
        {"role": "user", "content": f"[Summary of earlier conversation: {summary}]"},
        {"role": "assistant", "content": "Understood, I have the context from our earlier conversation."},
        *recent,
    ]


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class Source(BaseModel):
    filename: str
    page: int


class RetrievedChunk(BaseModel):
    rank: int
    filename: str
    page: int
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_log_id: str
    refused: bool
    needs_clarification: bool = False
    retrieval_latency_ms: int
    embed_latency_ms: int
    search_latency_ms: int
    generation_latency_ms: int
    retrieved_chunks: list[RetrievedChunk]


class FeedbackRequest(BaseModel):
    feedback: int  # 1 = thumbs up, -1 = thumbs down


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    return store.list_documents()


@app.get("/api/logs")
def list_logs(limit: int = 20):
    return store.list_query_logs(limit)


@app.get("/api/stats")
def stats():
    return store.get_stats()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = _apply_history_window(request.history, llm_client)

    clarification = check_needs_clarification(request.message, history, llm_client)
    if clarification:
        return ChatResponse(
            answer=clarification,
            sources=[],
            query_log_id="",
            refused=False,
            needs_clarification=True,
            retrieval_latency_ms=0,
            embed_latency_ms=0,
            search_latency_ms=0,
            generation_latency_ms=0,
            retrieved_chunks=[],
        )

    retrieval_start = time.monotonic()
    retrieval = retriever.retrieve_with_timing(request.message)
    results = retrieval.results
    retrieval_latency_ms = int((time.monotonic() - retrieval_start) * 1000)

    context = "\n\n".join(
        f"[{doc.filename}, page {chunk.page_number}]\n{chunk.content}" for chunk, doc in results
    )

    generation_start = time.monotonic()
    answer = "".join(llm_client.chat_stream(request.message, context, history=history))
    generation_latency_ms = int((time.monotonic() - generation_start) * 1000)

    seen: set[tuple[str, int]] = set()
    sources = []
    for chunk, doc in results:
        key = (doc.filename, chunk.page_number)
        if key not in seen:
            seen.add(key)
            sources.append(Source(filename=doc.filename, page=chunk.page_number))

    retrieved_chunks = [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "filename": doc.filename,
            "page": chunk.page_number,
            "rank": rank,
        }
        for rank, (chunk, doc) in enumerate(results)
    ]
    refused = is_refusal(answer)
    cited_sources = extract_citations(answer, results)

    query_log_id = store.log_query(
        question=request.message,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        cited_sources=cited_sources,
        refused=refused,
        retrieval_latency_ms=retrieval_latency_ms,
        embed_latency_ms=retrieval.embed_latency_ms,
        search_latency_ms=retrieval.search_latency_ms,
        generation_latency_ms=generation_latency_ms,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        query_log_id=query_log_id,
        refused=refused,
        retrieval_latency_ms=retrieval_latency_ms,
        embed_latency_ms=retrieval.embed_latency_ms,
        search_latency_ms=retrieval.search_latency_ms,
        generation_latency_ms=generation_latency_ms,
        retrieved_chunks=[
            RetrievedChunk(rank=rank, filename=doc.filename, page=chunk.page_number, preview=chunk.content[:160])
            for rank, (chunk, doc) in enumerate(results)
        ],
    )


@app.post("/api/feedback/{query_log_id}")
def feedback(query_log_id: str, request: FeedbackRequest):
    if request.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
    store.set_feedback(query_log_id, request.feedback)
    return {"status": "ok"}
