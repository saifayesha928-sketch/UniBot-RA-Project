"""Shared pre-retrieval query pipeline.

Provides the same classification, hint resolution, and query rewriting logic
that the /query API route implements inline. Currently consumed by the
evaluation runner; the route keeps its own copy because it interleaves
shadow-mode logging and caching between pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING, Protocol

from unibot.retrieval.query_classification import (
    classify_query,
    extract_secondary_hints,
    reconcile_classification,
    resolve_effective_source_class_hint,
)

if TYPE_CHECKING:
    from unibot.retrieval.query_classification import QueryClassification
    from unibot.retrieval.query_rewriter import RewriteResult


class _Classifier(Protocol):
    def classify(self, query_text: str) -> QueryClassification: ...


class _Rewriter(Protocol):
    def rewrite(self, query_text: str) -> RewriteResult: ...


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    original_query: str
    retrieval_query: str
    effective_source_class_hint: str | None
    record_type_hint: str | None
    hint_is_user_provided: bool
    abstain_immediately: bool
    abstain_reason: str
    secondary_hints: tuple[tuple[str | None, str | None], ...] = ()


def prepare_query(
    *,
    query_text: str,
    requested_source_class_hint: str | None,
    classifier_backend: str = "keyword",
    semantic_classifier: _Classifier | None = None,
    query_rewriter: _Rewriter | None = None,
) -> PreparedQuery:
    """Run classification, hint resolution, and query rewriting."""
    keyword_classification = classify_query(query_text)

    secondary_hints: tuple[tuple[str | None, str | None], ...] = ()

    if classifier_backend == "semantic" and semantic_classifier is not None:
        classify_multi = getattr(semantic_classifier, "classify_with_secondary", None)
        if callable(classify_multi):
            raw_semantic, secondary_hints = classify_multi(query_text)
        else:
            raw_semantic = semantic_classifier.classify(query_text)
            secondary_hints = extract_secondary_hints(query_text, primary=raw_semantic)
        classification = reconcile_classification(keyword_classification, raw_semantic)
    else:
        classification = keyword_classification
        secondary_hints = extract_secondary_hints(query_text, primary=classification)

    if classification.abstain_immediately:
        return PreparedQuery(
            original_query=query_text,
            retrieval_query=query_text,
            effective_source_class_hint=None,
            record_type_hint=None,
            hint_is_user_provided=False,
            abstain_immediately=True,
            abstain_reason=classification.reason,
            secondary_hints=(),
        )

    effective_source_class_hint = resolve_effective_source_class_hint(
        classified_hint=classification.source_class_hint,
        requested_hint=requested_source_class_hint,
    )

    retrieval_query = query_text
    if query_rewriter is not None:
        rewrite_result = query_rewriter.rewrite(query_text)
        if rewrite_result.rewritten_query != rewrite_result.original_query:
            retrieval_query = rewrite_result.rewritten_query

    return PreparedQuery(
        original_query=query_text,
        retrieval_query=retrieval_query,
        effective_source_class_hint=effective_source_class_hint,
        record_type_hint=classification.record_type_hint,
        hint_is_user_provided=requested_source_class_hint is not None,
        abstain_immediately=False,
        abstain_reason="",
        secondary_hints=secondary_hints,
    )
