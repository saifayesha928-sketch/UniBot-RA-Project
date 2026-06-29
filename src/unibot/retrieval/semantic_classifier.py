from __future__ import annotations

import math

from unibot.retrieval.query_classification import (
    QueryClass,
    QueryClassification,
    extract_record_type_hint,
    _OUT_OF_SCOPE_TERMS,
    _IN_SCOPE_TERMS,
)

# Exemplar queries for each QueryClass.  6-10 per class.
_EXEMPLARS: dict[QueryClass, list[str]] = {
    QueryClass.ENTITY_LOOKUP: [
        "What departments does the university have?",
        "Where is the university campus located?",
        "What programs does the university offer?",
        "Tell me about the computer science department",
        "What is the student portal?",
        "How do I contact the university administration?",
        "What are the library hours?",
        "What student services does the university provide?",
    ],
    QueryClass.POLICY_OR_THRESHOLD: [
        "What is the attendance policy?",
        "What is the minimum CGPA requirement?",
        "What are the grading rules?",
        "What is the policy for academic probation?",
        "What regulations apply to exam retakes?",
        "What is the threshold for passing a course?",
        "What are the rules for course withdrawal?",
        "What is the academic integrity policy?",
    ],
    QueryClass.FEE_OR_ADMISSIONS_CYCLE: [
        "What are the tuition fees for BS Computer Science?",
        "What is the fee structure for MS programs?",
        "When is the admission deadline for fall intake?",
        "What are the eligibility criteria for admission?",
        "How do I apply for a scholarship?",
        "What is the merit criteria for BS programs?",
        "When does the spring admission cycle open?",
        "What fees are required at the time of admission?",
    ],
    QueryClass.FACULTY_EXPERTISE_OR_PUBLICATION: [
        "What are the research interests of the faculty?",
        "List publications by Dr. Ahmed in machine learning",
        "Who are the professors in the CS department?",
        "What expertise does the AI lab faculty have?",
        "Show me faculty publications in data science",
        "Which professors work on natural language processing?",
        "What research groups exist at the university?",
        "Faculty expertise in computer vision",
    ],
    QueryClass.NEWS_OR_EVENT: [
        "What events are happening at the university this month?",
        "When is the next convocation?",
        "Are there any upcoming seminars?",
        "What workshops are available for students?",
        "University conference announcements",
        "Latest news from campus",
    ],
    QueryClass.OUT_OF_SCOPE: [
        "What is the weather forecast for tomorrow?",
        "What is the current stock price of Apple?",
        "Give me a recipe for biryani",
        "What is the score of the cricket match?",
        "Tell me about the latest NBA game",
        "What is the bitcoin price today?",
    ],
}

# Map QueryClass to default hints for semantic classification.
_CLASS_HINTS: dict[QueryClass, tuple[str | None, str | None]] = {
    QueryClass.ENTITY_LOOKUP: (None, None),
    QueryClass.POLICY_OR_THRESHOLD: ("policy", None),
    QueryClass.FEE_OR_ADMISSIONS_CYCLE: (None, None),
    QueryClass.FACULTY_EXPERTISE_OR_PUBLICATION: ("faculty", None),
    QueryClass.NEWS_OR_EVENT: ("news_event", "news_event"),
    QueryClass.OUT_OF_SCOPE: (None, None),
}

_EPSILON = 1e-10


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < _EPSILON or norm_b < _EPSILON:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _is_out_of_scope_fast_path(query_text: str) -> bool:
    """Keyword fast-path for obvious out-of-scope terms."""
    normalized = query_text.strip().lower()
    return (
        any(term in normalized for term in _OUT_OF_SCOPE_TERMS)
        and not any(term in normalized for term in _IN_SCOPE_TERMS)
    )


_OUT_OF_SCOPE_CLASSIFICATION = QueryClassification(
    query_class=QueryClass.OUT_OF_SCOPE,
    source_class_hint=None,
    record_type_hint=None,
    abstain_immediately=True,
    reason="Keyword fast-path: query contains out-of-scope terms.",
)


def _build_primary(
    best_class: QueryClass,
    best_score: float,
    query_text: str,
    threshold: float,
) -> QueryClassification:
    """Build a QueryClassification from the best-scoring class."""
    if best_score < threshold:
        return QueryClassification(
            query_class=QueryClass.ENTITY_LOOKUP,
            source_class_hint=None,
            record_type_hint=None,
            abstain_immediately=False,
            reason="Low-confidence semantic match; falling back to broad routing.",
        )

    if best_class == QueryClass.OUT_OF_SCOPE:
        return QueryClassification(
            query_class=best_class,
            source_class_hint=None,
            record_type_hint=None,
            abstain_immediately=True,
            reason=f"Semantic classification: {best_class.value} (score={best_score:.3f}).",
        )

    src_hint, rec_hint = _CLASS_HINTS.get(best_class, (None, None))
    if rec_hint is None:
        rec_hint = extract_record_type_hint(query_text, best_class)
    return QueryClassification(
        query_class=best_class,
        source_class_hint=src_hint,
        record_type_hint=rec_hint,
        abstain_immediately=False,
        reason=f"Semantic classification: {best_class.value} (score={best_score:.3f}).",
    )


class SemanticQueryClassifier:
    """Cosine-similarity query classifier using dense embeddings.

    Embeds a fixed set of exemplar queries at initialization, then classifies
    new queries by finding the nearest exemplar via cosine similarity.
    Falls back to broad ENTITY_LOOKUP routing when confidence is below threshold.
    """

    def __init__(
        self,
        *,
        dense_embedding_provider: object,
        threshold: float = 0.6,
    ) -> None:
        self._dense_provider = dense_embedding_provider
        self._threshold = threshold
        self._exemplar_classes: list[QueryClass] = []
        self._exemplar_vectors: list[tuple[float, ...]] = []
        self._embed_exemplars()

    def _embed_exemplars(self) -> None:
        all_texts: list[str] = []
        all_classes: list[QueryClass] = []
        for query_class, exemplars in _EXEMPLARS.items():
            for text in exemplars:
                all_texts.append(text)
                all_classes.append(query_class)

        embed_batch = getattr(self._dense_provider, "embed_dense_query_batch", None)
        if callable(embed_batch):
            vectors = embed_batch(all_texts)
            for query_class, vector in zip(all_classes, vectors):
                self._exemplar_classes.append(query_class)
                self._exemplar_vectors.append(tuple(vector))
        else:
            embed_single = getattr(self._dense_provider, "embed_dense_query", None)
            if not callable(embed_single):
                raise TypeError(
                    "dense_embedding_provider must have embed_dense_query or "
                    "embed_dense_query_batch method"
                )
            for query_class, text in zip(all_classes, all_texts):
                vector = embed_single(text)
                self._exemplar_classes.append(query_class)
                self._exemplar_vectors.append(tuple(vector))

    def _embed_query(self, query_text: str) -> tuple[float, ...]:
        """Embed a single query string using the dense provider."""
        embed_single = getattr(self._dense_provider, "embed_dense_query", None)
        if not callable(embed_single):
            raise TypeError("dense_embedding_provider must have embed_dense_query method")
        return tuple(embed_single(query_text))

    def _score_classes(
        self, query_vector: tuple[float, ...],
    ) -> dict[QueryClass, float]:
        """Compute best similarity score per class across all exemplars."""
        class_scores: dict[QueryClass, list[float]] = {}
        for i, exemplar_vec in enumerate(self._exemplar_vectors):
            score = _cosine_similarity(query_vector, exemplar_vec)
            qclass = self._exemplar_classes[i]
            class_scores.setdefault(qclass, []).append(score)
        return {qclass: max(scores) for qclass, scores in class_scores.items()}

    def classify(self, query_text: str) -> QueryClassification:
        if _is_out_of_scope_fast_path(query_text):
            return _OUT_OF_SCOPE_CLASSIFICATION

        query_vector = self._embed_query(query_text)
        class_best = self._score_classes(query_vector)

        best_class = max(class_best, key=class_best.get)  # type: ignore[arg-type]
        best_score = class_best[best_class]
        return _build_primary(best_class, best_score, query_text, self._threshold)

    def classify_with_secondary(
        self,
        query_text: str,
    ) -> tuple[QueryClassification, tuple[tuple[str | None, str | None], ...]]:
        """Classify query and return secondary intent hints for multi-intent queries.

        Returns the primary QueryClassification (same as classify()) plus a tuple
        of (source_class_hint, record_type_hint) pairs for all additional intent
        classes that scored above the threshold.

        Single embedding call, single similarity scan — no extra latency.
        """
        if _is_out_of_scope_fast_path(query_text):
            return _OUT_OF_SCOPE_CLASSIFICATION, ()

        query_vector = self._embed_query(query_text)
        class_best = self._score_classes(query_vector)

        best_class = max(class_best, key=class_best.get)  # type: ignore[arg-type]
        best_score = class_best[best_class]
        primary = _build_primary(best_class, best_score, query_text, self._threshold)

        # Secondary: all other classes above threshold, excluding primary and OUT_OF_SCOPE
        primary_pair = (primary.source_class_hint, primary.record_type_hint)
        secondary: list[tuple[str | None, str | None]] = []
        seen: set[tuple[str | None, str | None]] = {primary_pair}

        for qclass, score in sorted(
            class_best.items(), key=lambda x: x[1], reverse=True
        ):
            if qclass == best_class:
                continue
            if qclass == QueryClass.OUT_OF_SCOPE:
                continue
            if score < self._threshold:
                continue
            src_hint, rec_hint = _CLASS_HINTS.get(qclass, (None, None))
            if rec_hint is None:
                rec_hint = extract_record_type_hint(query_text, qclass)
            pair = (src_hint, rec_hint)
            if pair in seen:
                continue
            seen.add(pair)
            secondary.append(pair)

        return primary, tuple(secondary)
