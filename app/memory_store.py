import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryItem:
    """Simple record representing a stored memory snippet."""

    session_id: str
    turn_index: int
    role: str
    text: str
    summary: str
    embedding: np.ndarray
    created_at: float
    outcome: Optional[str] = None
    tags: Optional[list[str]] = None

    def format_for_prompt(self) -> str:
        """Return a concise string for inclusion in the system prompt."""
        return f"[{self.turn_index}] {self.summary}"


class MemoryStore:
    """Lightweight vector store for conversation memories."""

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        max_items_per_session: Optional[int] = None,
    ):
        self.embedding_model = embedding_model
        self.max_items_per_session = max_items_per_session
        self._items: List[MemoryItem] = []

    def add(
        self,
        session_id: str,
        role: str,
        text: str,
        summary: Optional[str] = None,
        turn_index: Optional[int] = None,
        outcome: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> MemoryItem:
        """Store a new memory entry and return it."""
        content = summary if summary else text
        embedding = self.embedding_model.encode(content)
        idx = (
            turn_index
            if turn_index is not None
            else self._next_turn_index(session_id=session_id)
        )
        item = MemoryItem(
            session_id=session_id,
            turn_index=idx,
            role=role,
            text=text,
            summary=summary if summary else text,
            embedding=embedding,
            created_at=time.time(),
            outcome=outcome,
            tags=tags,
        )
        self._items.append(item)
        self._prune(session_id=session_id)
        self._after_add()
        return item

    def search(self, session_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        """Return top-k similar memories for the session."""
        session_items = [i for i in self._items if i.session_id == session_id]
        if not session_items:
            return []

        query_embedding = self.embedding_model.encode(query)
        embedding_matrix = np.vstack([i.embedding for i in session_items])

        scores = cosine_similarity(query_embedding.reshape(1, -1), embedding_matrix)[0]
        top_indices = scores.argsort()[-k:][::-1]
        return [session_items[i] for i in top_indices]

    def search_with_scores(
        self, session_id: str, query: str, k: int = 5
    ) -> List[tuple[MemoryItem, float]]:
        """
        Return top-k similar memories and their cosine scores.
        Useful for fast-answer shortcuts that need a confidence threshold.
        """
        session_items = [i for i in self._items if i.session_id == session_id]
        if not session_items:
            return []

        query_embedding = self.embedding_model.encode(query)
        embedding_matrix = np.vstack([i.embedding for i in session_items])
        scores = cosine_similarity(query_embedding.reshape(1, -1), embedding_matrix)[0]
        top_indices = scores.argsort()[-k:][::-1]
        return [(session_items[i], float(scores[i])) for i in top_indices]

    def search_by_outcome(
        self, query: str, outcome: Optional[str], k: int = 5
    ) -> List[MemoryItem]:
        """Return top-k memories filtered by outcome label."""
        if outcome:
            pool = [i for i in self._items if i.outcome == outcome]
        else:
            pool = list(self._items)

        if not pool:
            return []

        query_embedding = self.embedding_model.encode(query)
        embedding_matrix = np.vstack([i.embedding for i in pool])
        scores = cosine_similarity(query_embedding.reshape(1, -1), embedding_matrix)[0]
        top_indices = scores.argsort()[-k:][::-1]
        return [pool[i] for i in top_indices]

    def search_best_practice(
        self,
        query: str,
        outcome: Optional[str],
        tags: Optional[list[str]] = None,
        k: int = 5,
    ) -> List[tuple[MemoryItem, float]]:
        """
        Return top-k best-practice memories as (item, score), filtered by outcome
        and optionally by tag overlap.
        """
        if outcome:
            pool = [i for i in self._items if i.outcome == outcome]
        else:
            pool = list(self._items)

        if tags:
            tag_set = set(t.strip().lower() for t in tags if t)
            pool = [
                i
                for i in pool
                if i.tags
                and tag_set.intersection({t.strip().lower() for t in i.tags if t})
            ]

        if not pool:
            return []

        query_embedding = self.embedding_model.encode(query)
        embedding_matrix = np.vstack([i.embedding for i in pool])
        scores = cosine_similarity(query_embedding.reshape(1, -1), embedding_matrix)[0]
        top_indices = scores.argsort()[-k:][::-1]
        return [(pool[i], float(scores[i])) for i in top_indices]

    def _next_turn_index(self, session_id: str) -> int:
        session_items = [i for i in self._items if i.session_id == session_id]
        if not session_items:
            return 0
        return max(i.turn_index for i in session_items) + 1

    def _prune(self, session_id: str) -> None:
        """Enforce max_items_per_session, keeping most recent."""
        if not self.max_items_per_session:
            return
        session_items = [i for i in self._items if i.session_id == session_id]
        if len(session_items) <= self.max_items_per_session:
            return
        # Keep the newest items and drop older ones
        session_items.sort(key=lambda i: i.created_at, reverse=True)
        keep_ids = {id(i) for i in session_items[: self.max_items_per_session]}
        self._items = [
            i
            for i in self._items
            if (i.session_id != session_id) or (id(i) in keep_ids)
        ]

    def _after_add(self) -> None:
        """Hook for subclasses to persist changes."""
        return


class JsonMemoryStore(MemoryStore):
    """File-backed memory store using JSON for simple persistence."""

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        file_path: str,
        max_items_per_session: Optional[int] = None,
    ):
        self.file_path = file_path
        dir_path = os.path.dirname(self.file_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        super().__init__(
            embedding_model=embedding_model,
            max_items_per_session=max_items_per_session,
        )
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, "r") as f:
            try:
                raw_items = json.load(f)
            except json.JSONDecodeError:
                raw_items = []
        for raw in raw_items:
            embedding = np.array(raw.get("embedding", []))
            item = MemoryItem(
                session_id=raw["session_id"],
                turn_index=raw["turn_index"],
                role=raw["role"],
                text=raw["text"],
                summary=raw["summary"],
                embedding=embedding,
                created_at=raw["created_at"],
                outcome=raw.get("outcome"),
                tags=raw.get("tags"),
            )
            self._items.append(item)

    def _after_add(self) -> None:
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        serializable = []
        for item in self._items:
            record = asdict(item)
            record["embedding"] = item.embedding.tolist()
            serializable.append(record)
        with open(self.file_path, "w") as f:
            json.dump(serializable, f)
