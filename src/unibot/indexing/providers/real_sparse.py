from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from unibot.indexing.embeddings import SparseVector

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class TokenSparseEmbeddingProvider:
    def embed_sparse(self, text: str) -> SparseVector:
        return self.embed_sparse_document(text)

    def embed_sparse_document(self, text: str) -> SparseVector:
        return _embed_sparse_tokens(text)

    def embed_sparse_query(self, text: str) -> SparseVector:
        return _embed_sparse_tokens(text)


def _embed_sparse_tokens(text: str) -> SparseVector:
    token_counts = Counter(_TOKEN_PATTERN.findall(text.lower()))
    if not token_counts:
        return SparseVector(indices=(1,), values=(1.0,))

    ordered_tokens = sorted(token_counts.items())
    indices = tuple(_token_index(token) for token, _count in ordered_tokens)
    values = tuple(round(1.0 + math.log1p(count), 6) for _token, count in ordered_tokens)
    return SparseVector(indices=indices, values=values)


def _token_index(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % 2_147_483_000) + 1
