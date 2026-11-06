from ragapp.eval.metrics import (
    answer_relevancy,
    citation_accuracy,
    context_entities_recall,
    context_utilization,
    faithfulness,
    noise_sensitivity,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DOC = "manual.pdf"


def test_recall_at_k_full_hit():
    retrieved = [(DOC, 1), (DOC, 5), (DOC, 9)]
    relevant = {(DOC, 5)}
    assert recall_at_k(retrieved, relevant) == 1.0


def test_recall_at_k_miss():
    retrieved = [(DOC, 1), (DOC, 2)]
    relevant = {(DOC, 5)}
    assert recall_at_k(retrieved, relevant) == 0.0


def test_precision_at_k():
    retrieved = [(DOC, 1), (DOC, 5), (DOC, 9), (DOC, 10)]
    relevant = {(DOC, 5), (DOC, 9)}
    assert precision_at_k(retrieved, relevant) == 0.5


def test_reciprocal_rank_first_hit():
    retrieved = [(DOC, 5), (DOC, 1)]
    relevant = {(DOC, 5)}
    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_reciprocal_rank_second_hit():
    retrieved = [(DOC, 1), (DOC, 5)]
    relevant = {(DOC, 5)}
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_no_hit():
    retrieved = [(DOC, 1), (DOC, 2)]
    relevant = {(DOC, 5)}
    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_citation_accuracy_none_when_no_citations():
    assert citation_accuracy([], [(DOC, 1)]) is None


def test_citation_accuracy_detects_fabricated_citation():
    cited = [(DOC, 1), (DOC, 99)]
    retrieved = [(DOC, 1), (DOC, 2)]
    assert citation_accuracy(cited, retrieved) == 0.5


def test_context_utilization():
    cited = [(DOC, 1)]
    retrieved = [(DOC, 1), (DOC, 2), (DOC, 3)]
    assert context_utilization(cited, retrieved) == 1 / 3


# ---------------------------------------------------------------------------
# faithfulness
# ---------------------------------------------------------------------------

def test_faithfulness_grounded_answer():
    context = ["Press the reset button on the dustbin compartment to restore factory settings."]
    answer = "Press the reset button on the dustbin compartment."
    assert faithfulness(answer, context) == 1.0


def test_faithfulness_ungrounded_sentence_lowers_score():
    context = ["Clean the filter weekly by tapping it over a bin."]
    # second sentence shares no content tokens with context
    answer = "Clean the filter weekly. Order fresh supplies from Amazon."
    score = faithfulness(answer, context)
    assert 0.0 < score < 1.0


def test_faithfulness_empty_answer():
    assert faithfulness("", ["some context here"]) == 1.0


# ---------------------------------------------------------------------------
# answer_relevancy
# ---------------------------------------------------------------------------

def test_answer_relevancy_high_overlap():
    score = answer_relevancy(
        "How do I clean the dustbin?",
        "Remove and clean the dustbin by tapping it over a waste bin.",
    )
    assert score > 0.5


def test_answer_relevancy_off_topic():
    score = answer_relevancy(
        "How do I clean the dustbin?",
        "Contact customer support for warranty information.",
    )
    assert score < 0.5


def test_answer_relevancy_empty_inputs():
    assert answer_relevancy("", "some answer") == 0.0
    assert answer_relevancy("some question", "") == 0.0


# ---------------------------------------------------------------------------
# noise_sensitivity
# ---------------------------------------------------------------------------

def test_noise_sensitivity_no_noise():
    relevant = {(DOC, 1), (DOC, 2)}
    cited = [(DOC, 1), (DOC, 2)]
    assert noise_sensitivity(cited, relevant) == 0.0


def test_noise_sensitivity_all_noise():
    relevant = {(DOC, 1)}
    cited = [(DOC, 5), (DOC, 9)]
    assert noise_sensitivity(cited, relevant) == 1.0


def test_noise_sensitivity_partial():
    relevant = {(DOC, 1)}
    cited = [(DOC, 1), (DOC, 5)]
    assert noise_sensitivity(cited, relevant) == 0.5


def test_noise_sensitivity_none_when_no_citations():
    assert noise_sensitivity([], {(DOC, 1)}) is None


# ---------------------------------------------------------------------------
# context_entities_recall
# ---------------------------------------------------------------------------

def test_context_entities_recall_full():
    chunks = ["Press dustbin button to reset factory settings"]
    answer = "Press the dustbin button to reset factory settings."
    assert context_entities_recall(answer, chunks) == 1.0


def test_context_entities_recall_partial():
    chunks = ["Press dustbin reset button restore factory settings"]
    answer = "Press the dustbin button."
    score = context_entities_recall(answer, chunks)
    assert 0.0 < score < 1.0


def test_context_entities_recall_empty_chunks():
    assert context_entities_recall("some answer", []) == 0.0
