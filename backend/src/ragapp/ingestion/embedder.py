import time

import httpx

from ragapp.config import settings

_BATCH_SIZE = 100
_MAX_RETRIES = 6
_RETRY_BASE = 2.0


class Embedder:
    """Calls OpenAI's /embeddings endpoint. Swap implementation to change
    embedding backend without touching ingestion/retrieval callers.

    Keeps one persistent httpx.Client per instance so repeated calls reuse
    the underlying HTTP connection instead of paying connection setup cost.
    Sends texts in batches of _BATCH_SIZE and retries 429s with exponential backoff.
    """

    def __init__(self, model: str | None = None):
        self.model = model or settings.openai_embedding_model
        self._client = httpx.Client(
            base_url=settings.openai_base_url,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=60,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            results.extend(self._embed_batch(batch))
        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            response = self._client.post(
                "/embeddings",
                json={"model": self.model, "input": texts},
            )
            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", _RETRY_BASE ** attempt))
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        response.raise_for_status()
        return []  # unreachable
