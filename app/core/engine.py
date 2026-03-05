import os
import re
import json
import numpy as np
from typing import List, Optional

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

def normalise_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

class PromptGuidance:
    """
    Loads soft behaviour Guidance and builds a compact per-turn breif
    that can be appeneded to the system instructions
    """

    def __init__(
            self,
            policy_path: Optional[str] = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(base_dir, "prompts")
        self.policy_path = policy_path or os.path.join(prompts_dir, "prompt_policy.md")

        self.policy_text = self._load_policy()

    def _load_policy(self) -> str:
        if not os.path.exists(self.policy_path):
            return ""
        with open(self.policy_path, "r", encoding="utf-8") as f:
            return f.read().strip()
        
    def build_style_brief(self, latest_user_prompt: str, history: List[dict]) -> str:
       # parameters are kept for interface compatability 
        _ = latest_user_prompt
        _ = history
        parts: List[str] = []

        if self.policy_text:
            parts.append("## OJ tool Interaction Guidance (soft)\n" + self.policy_text)

        return "\n\n".join(parts).strip()
    
    def compose_system_instruction(
            self,
            base_system_instruction: str,
            latest_user_prompt:str,
            history: List[dict],
        ) -> str:
        style_brief = self.build_style_brief(latest_user_prompt, history)

        if not style_brief:
            return base_system_instruction
        
        priority = (
            "## Priority Order\n"
            "1) Saftey and policy compliance\n"
            "2) Correctness and ground answers\n"
            "3) User task Completion\n"
            "4) Style alignment from guidance\n"
        )
        return f"{base_system_instruction}\n\n{priority}\n\n{style_brief}".strip()
    
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

        normalised_query = normalise_query(query)

        for row in interactions:
            if normalise_query(row["query"]) == normalised_query:
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