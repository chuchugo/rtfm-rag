# Roadmap

## Done

- [x] Repo scaffolding, specs, CLAUDE.md.

## Phase 1 — local MVP

- [x] Install prerequisites (Python, Node, Docker, Ollama) — see [README.md](../../README.md).
- [x] `docker compose up` — Postgres + pgvector running, schema created.
- [x] Ingestion CLI: PDF → chunks → embeddings → DB (Tesla Model 3 manual, 1498 chunks).
- [x] Retrieval: query → top-k chunks, verified against real questions.
- [x] FastAPI `/api/chat` endpoint wired to Ollama, grounded in retrieved chunks.
- [x] Minimal React chat UI hitting the API, showing answers + sources.
- [x] Eval/monitoring: `query_logs` table, Recall@K/MRR/citation-accuracy harness
  (`python -m ragapp.cli eval`), Stats panel + per-answer Details panel in the UI —
  see [06-evaluation.md](06-evaluation.md).
- [x] Performance pass: connection pooling, persistent HTTP clients, model swap
  (llama3.1 → llama3.2:3b), `keep_alive`, IPv4 host fix, full GPU offload via
  `num_ctx`. Retrieval 3-4s → ~200ms, generation 17-30s → 5-8s.
- [x] **Needed by rtfm-hub**: `VectorStore.search()` takes an optional
  `document_ids` filter; ingestion factored out of `cli.py` into
  `ingestion/service.py`'s `ingest_pdf()` (raises `IngestionError`, not
  `SystemExit` — safe to call as a library).
- [x] **Needed by rtfm-hub**: pluggable LLM provider. `ragapp.llm.base.LLMProvider`
  is the interface (`chat_stream(user_message, context) -> Iterator[str]`);
  `providers/ollama.py`, `providers/openai_compat.py`, `providers/anthropic.py`
  implement it; `llm/factory.py`'s `create_llm_client()` builds whichever
  `LLM_PROVIDER` env var says (default `ollama`). "Bring your own LLM" is now a
  config choice, not a rewrite, for both this app and rtfm-hub.

## Phase 2 — polish

- [ ] Ingest multiple manuals, verify retrieval quality across documents.
- [ x] Basic auth if exposed beyond localhost.
- [x ] Fill out `backend/eval/questions.yaml` with a real labeled question set
  (currently just a template) and run Recall@K/MRR for real.
- [ ] Fix refusal-detection heuristic gaps found during model benchmarking
  (`heuristics.py` misses phrasings like "do not contain" / "not aware of any
  information" — only catches "does not contain").
- [ ] Benchmark `llama3.2:1b` — smaller/faster than the current `llama3.2:3b`
  default; worth trying if answer latency matters more than depth for some
  use cases. Trade-off is untested quality loss at this size.

- [x] add guardrail in prompts: type of product in scope, type of quetsion in scope, limitation on actions: only suggestion to the user, no direct actions outside of the scope of the manuals.
- [ ] compare dif model
- [x] if eval can cover more types of question, other types of eval include Faithfulness, Answer Relevancy, Noise Sensitivity, Context Entities Recall, 
- [ ] multiturn : memory + build up the context

## Phase 3 — future / exploratory

- [ ] Independent cloud deployment for demos.
- [ ] Hybrid search / re-ranking if retrieval quality needs it.

Note: Home Assistant integration is being built as rtfm-hub (a separate repo that
imports this one as a library), not here — see `../rtfm-hub/docs/specs/`.
