import json
import os
import re
import numpy as np

from dataclasses import dataclass
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer

def _normalise_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

@dataclass
class CacheMatch:
    answer: str
    score: float
    matched_query: str

class CAGQueryCache:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lookup(self, query: str, threshold: float = 0.92) -> Optional[CacheMatch]:
        interactions = self._load_interactions()
        if not interactions:
            return None

        normalised_query = _normalise_query(query)

        for row in interactions:
            if _normalise_query(row["query"]) == normalised_query:
                return CacheMatch(answer=row["answer"], score=1.0, matched_query=row["query"])

        cache_queries = [row["query"] for row in interactions]
        all_queries = cache_queries + [query]

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        vectors = vectorizer.fit_transform(all_queries)
        cache_vectors = vectors[:-1]
        query_vector = vectors[-1]

        similarities = (cache_vectors @ query_vector.T).toarray().ravel()
        if similarities.size == 0:
            return None

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < threshold:
            return None

        return CacheMatch(
            answer=interactions[best_idx]["answer"],
            score=best_score,
            matched_query=interactions[best_idx]["query"],
        )

    def _load_interactions(self) -> list[dict]:
        if not os.path.exists(self.file_path):
            return []

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(payload, list):
            return []

        cleaned = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            answer = item.get("answer")
            if isinstance(query, str) and query.strip() and isinstance(answer, str) and answer.strip():
                cleaned.append({"query": query, "answer": answer})

        return cleaned