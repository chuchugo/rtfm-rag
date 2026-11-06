"""Retrieval/generation metric primitives.

Each function scores a single query; callers average across queries to get
an aggregate (e.g. `mean(recall_at_k(...) for q in questions)`).
"""

import re

PageRef = tuple[str, int]  # (filename, page_number)

# Common English function words that carry no domain signal.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "and", "or",
    "of", "for", "with", "that", "this", "are", "was", "be", "as", "by",
    "from", "not", "but", "if", "you", "can", "will", "do", "i", "my",
    "how", "what", "when", "where", "which", "your", "have", "has", "its",
    "also", "been", "would", "could", "should", "then", "than", "about",
    "into", "them", "they", "their", "there", "these", "those", "just",
    "more", "some", "such", "each", "only", "other", "make", "made",
})


def _content_tokens(text: str) -> set[str]:
    """Lowercase alphabetic tokens of length ≥ 4 that are not stopwords."""
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def recall_at_k(retrieved: list[PageRef], relevant: set[PageRef]) -> float:
    """Fraction of the relevant pages that showed up anywhere in the retrieved set."""
    if not relevant:
        return 0.0
    hit = {page for page in retrieved if page in relevant}
    return len(hit) / len(relevant)


def precision_at_k(retrieved: list[PageRef], relevant: set[PageRef]) -> float:
    """Fraction of retrieved pages that were actually relevant."""
    if not retrieved:
        return 0.0
    hit = [page for page in retrieved if page in relevant]
    return len(hit) / len(retrieved)


def reciprocal_rank(retrieved: list[PageRef], relevant: set[PageRef]) -> float:
    """1/rank of the first relevant page in `retrieved` (1-indexed), else 0."""
    for rank, page in enumerate(retrieved, start=1):
        if page in relevant:
            return 1 / rank
    return 0.0


def citation_accuracy(cited: list[PageRef], retrieved: list[PageRef]) -> float | None:
    """Fraction of cited pages that were actually among the retrieved pages.

    Catches fabricated citations (the model naming a page it was never shown).
    Returns None when the answer made no citations to score — callers should
    exclude these from the aggregate rather than counting them as 0 or 1.
    """
    if not cited:
        return None
    retrieved_set = set(retrieved)
    hit = [page for page in cited if page in retrieved_set]
    return len(hit) / len(cited)


def context_utilization(cited: list[PageRef], retrieved: list[PageRef]) -> float:
    """Fraction of retrieved pages that made it into the answer's citations.

    Low values suggest top_k is fetching more than the model actually needs.
    """
    if not retrieved:
        return 0.0
    cited_set = set(cited)
    hit = [page for page in retrieved if page in cited_set]
    return len(hit) / len(set(retrieved))


# ---------------------------------------------------------------------------
# RAGAS-inspired generation-quality metrics (heuristic, no LLM judge needed)
# ---------------------------------------------------------------------------

def faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Fraction of answer sentences grounded in the retrieved context.

    A sentence is "grounded" when it shares at least 2 content tokens with
    the concatenated retrieved chunks. Sentences with no such overlap are
    treated as potentially hallucinated. Returns 1.0 when there are no
    sentences to score (e.g. empty answer).
    """
    sentences = _sentences(answer)
    if not sentences:
        return 1.0
    context_tokens = _content_tokens(" ".join(context_chunks))
    grounded = sum(
        1 for s in sentences if len(_content_tokens(s) & context_tokens) >= 2
    )
    return grounded / len(sentences)


def answer_relevancy(question: str, answer: str) -> float:
    """Overlap coefficient between question tokens and answer tokens.

    Measures whether the answer addresses the same topic as the question.
    Returns 0.0 when either side has no content tokens.
    """
    q_tokens = _content_tokens(question)
    a_tokens = _content_tokens(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / min(len(q_tokens), len(a_tokens))


def noise_sensitivity(cited: list[PageRef], relevant: set[PageRef]) -> float | None:
    """Fraction of cited pages that are outside the labeled-relevant set.

    High values mean the model is citing noisy (irrelevant) chunks. Returns
    None when there are no citations — callers should exclude these rather
    than counting them as 0 or 1.
    """
    if not cited:
        return None
    noisy = [page for page in cited if page not in relevant]
    return len(noisy) / len(cited)


def context_entities_recall(answer: str, relevant_chunks: list[str]) -> float:
    """Fraction of content tokens from the relevant chunks that appear in the answer.

    Measures how much of the useful context the model actually used. Returns
    0.0 when the relevant chunks have no content tokens to score against.
    """
    entity_tokens = _content_tokens(" ".join(relevant_chunks))
    if not entity_tokens:
        return 0.0
    return len(entity_tokens & _content_tokens(answer)) / len(entity_tokens)
