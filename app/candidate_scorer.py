import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CandidateConfig:
    ambiguity_similarity_gap: float = 0.05
    strong_candidate_threshold: float = 0.35
    confident_score_threshold: float = 0.55
    confident_margin_threshold: float = 0.15
    weak_candidate_floor: float = 0.25
    top_k: int = 3


class CandidateScorer:
    """
    Encapsulates candidate parsing and scoring so the agent stays focused
    on conversation flow.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        chunk_data: Sequence[str],
        embedding_model: SentenceTransformer,
        config: CandidateConfig,
    ) -> None:
        self.embeddings = embeddings
        self.chunk_data = chunk_data
        self.embedding_model = embedding_model
        self.config = config

    def parse_chunk_to_record(self, chunk: str) -> Dict[str, str]:
        """Parse a text chunk into structured fields for disambiguation."""
        patterns = {
            "uid": r"The unique id is\s+(.*?)\.",
            "service_name": r"The service name is\s+(.*?)\.",
            "department": r"The department is\s+(.*?)\.",
            "phone_number": r"The phone number is\s+(.*?)\.",
            "topic": r"The topic is\s+(.*?)\.",
            "user_type": r"The user type is\s+(.*?)\.",
            "tags": r"The tags are\s+(.*?)\.",
            "url": r"The url is\s+(.*?)\.",
            "description": r"The description is\s+(.*?)\.",
        }
        record: Dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, chunk, re.IGNORECASE | re.DOTALL)
            record[key] = match.group(1).strip() if match else ""
        return record

    def _tag_description_bonus(self, candidate: Dict[str, str], user_query: str) -> float:
        """
        Lightweight lexical bonus when tags/description terms appear in the query.
        Keeps bonuses small to avoid overpowering embedding similarity.
        """
        query_lower = user_query.lower()
        bonus = 0.0

        tags = candidate.get("tags", "")
        tag_tokens = [t.strip().lower() for t in re.split(r"[,/;|]", tags) if t.strip()]
        for token in tag_tokens:
            if token and token in query_lower:
                bonus += 0.02

        description = candidate.get("description", "")
        desc_tokens = [w for w in re.findall(r"[a-zA-Z]{4,}", description.lower())]
        for token in set(desc_tokens):
            if token in query_lower:
                bonus += 0.005

        return min(bonus, 0.05)

    def get_top_candidates(self, user_query: str, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return top-N candidates with scores for ambiguity detection."""
        if self.embeddings is None:
            return []

        query_embedding = self.embedding_model.encode(user_query)
        similarity_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]

        top_n = top_n or self.config.top_k
        all_candidates: List[Dict[str, Any]] = []
        for idx, base_score in enumerate(similarity_scores):
            record = self.parse_chunk_to_record(self.chunk_data[idx])
            bonus = self._tag_description_bonus(record, user_query)
            record["score"] = float(base_score + bonus)
            record["base_score"] = float(base_score)
            record["bonus"] = float(bonus)
            if record["score"] >= self.config.weak_candidate_floor:
                all_candidates.append(record)

        ranked = sorted(all_candidates, key=lambda c: c["score"], reverse=True)
        return ranked[:top_n]

    def needs_disambiguation(self, candidates: List[Dict[str, Any]]) -> bool:
        """Check whether multiple close-scoring candidates could be chosen."""
        if len(candidates) < 2:
            return False

        top_score = candidates[0].get("score", 0.0)
        second_score = candidates[1].get("score", 0.0)
        strong_candidates = [c for c in candidates if c.get("score", 0.0) >= self.config.strong_candidate_threshold]
        distinct_services = {
            (c.get("service_name", ""), c.get("department", ""), c.get("user_type", ""))
            for c in candidates
            if c.get("service_name")
        }

        close_scores = abs(top_score - second_score) <= self.config.ambiguity_similarity_gap
        has_multiple_strong = len(strong_candidates) >= 2
        return close_scores and has_multiple_strong and len(distinct_services) > 1

    def build_confidence_hint(self, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """
        Flag a confident single match to short-circuit clarifications.
        """
        if not candidates:
            return None

        top_score = candidates[0].get("score", 0.0)
        second_score = candidates[1].get("score", 0.0) if len(candidates) > 1 else 0.0
        margin = top_score - second_score

        strong_and_clear_margin = (
            top_score >= self.config.confident_score_threshold
            and margin >= self.config.confident_margin_threshold
        )
        single_strong_candidate = len(candidates) == 1 and top_score >= self.config.confident_score_threshold

        if not (strong_and_clear_margin or single_strong_candidate):
            return None

        candidate = candidates[0]
        service = candidate.get("service_name") or "one service"
        dept = candidate.get("department")
        details = [f"Single strong match: {service}"]
        if dept:
            details.append(f"department: {dept}")
        details.append(f"score={top_score:.2f}, margin={margin:.2f}")
        details.append("Skip clarifications and call the tool immediately with the user's query.")
        return ". ".join(details)

    def _candidate_regions(self, candidate: Dict[str, str]) -> List[str]:
        """Extract region labels from a candidate using loose heuristics."""
        region_tokens = [
            ("england", ["england", "english", "dfe", "uk"]),
            ("northern ireland", ["ni", "northern ireland"]),
            ("scotland", ["scotland", "scottish"]),
            ("wales", ["wales", "welsh"]),
        ]
        text = " ".join(
            [
                candidate.get("service_name", "") or "",
                candidate.get("department", "") or "",
                candidate.get("tags", "") or "",
                candidate.get("description", "") or "",
            ]
        ).lower()
        hits: List[str] = []
        for label, synonyms in region_tokens:
            if any(token in text for token in synonyms):
                hits.append(label)
        return hits

    def select_candidate_from_clarification(
        self, user_text: str, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Choose the best candidate based on the user's clarification text."""
        if not candidates:
            return None
        user_lower = user_text.lower()

        region_scores = []
        for cand in candidates:
            regions = self._candidate_regions(cand)
            score = 0
            for region in regions:
                if region in user_lower:
                    score += 2
            region_scores.append(score)

        best_region_idx = None
        if any(score > 0 for score in region_scores):
            best_region_idx = max(range(len(candidates)), key=lambda i: region_scores[i])

        if best_region_idx is not None and region_scores[best_region_idx] > 0:
            return candidates[best_region_idx]

        def overlap_score(cand: Dict[str, str]) -> int:
            text = " ".join(
                [
                    cand.get("service_name", ""),
                    cand.get("department", ""),
                    cand.get("tags", ""),
                    cand.get("description", ""),
                ]
            ).lower()
            words = set(re.findall(r"[a-zA-Z]{3,}", text))
            return sum(1 for w in words if w in user_lower)

        scored = sorted(
            [(overlap_score(c), idx, c) for idx, c in enumerate(candidates)],
            key=lambda x: (x[0], candidates[x[1]].get("score", 0.0)),
            reverse=True,
        )
        top_score, _, top_candidate = scored[0]
        return top_candidate if top_score > 0 else candidates[0]

    def build_disambiguation_question(
        self,
        candidates: List[Dict[str, Any]],
        slots: Dict[str, Optional[str]],
    ) -> Optional[str]:
        """Craft a concise question to pick between close matches."""
        if not self.needs_disambiguation(candidates):
            return None

        slot_priority = ["service_name", "department", "user_type", "tags"]
        missing_slots = [k for k, v in slots.items() if not v]

        distinct_values: Dict[str, List[str]] = {}
        for slot in slot_priority:
            seen = []
            for candidate in candidates[:3]:
                value = candidate.get(slot, "")
                if value:
                    clean_value = value.strip()
                    if clean_value and clean_value not in seen:
                        seen.append(clean_value)
            distinct_values[slot] = seen

        target_slot = next(
            (slot for slot in slot_priority if slot in missing_slots and len(distinct_values.get(slot, [])) > 1),
            None,
        )
        if target_slot is None:
            target_slot = next((slot for slot in slot_priority if len(distinct_values.get(slot, [])) > 1), None)

        if target_slot:
            examples = " or ".join(distinct_values[target_slot][:2])
            region_tokens = [
                ("england", ["england", "english", "dfe"]),
                ("Northern Ireland", ["ni", "northern ireland"]),
                ("Scotland", ["scotland", "scottish"]),
                ("Wales", ["wales", "welsh"]),
                ("UK-wide", ["uk", "united kingdom"]),
            ]
            region_hits: List[str] = []
            for candidate in candidates[:3]:
                text = " ".join(
                    [
                        candidate.get("service_name", "") or "",
                        candidate.get("department", "") or "",
                        candidate.get("tags", "") or "",
                    ]
                ).lower()
                for label, synonyms in region_tokens:
                    if label in region_hits:
                        continue
                    if any(token in text for token in synonyms):
                        region_hits.append(label)

            if len(region_hits) > 1:
                top_regions = " or ".join(region_hits[:2])
                return f"Are you based in {top_regions}? That decides which contact to give you."

            if target_slot == "department":
                return f"Which department does this relate to? For example: {examples}."
            if target_slot == "user_type":
                return f"Are you asking as {examples}? That helps me pick the right contact."
            if target_slot == "tags":
                return f"Which topic best fits this request ({examples})?"
            if target_slot == "service_name":
                return f"Which service name matches best: {examples}?"

        options = []
        for candidate in candidates[:2]:
            service = candidate.get("service_name", "").strip() or "this service"
            dept = candidate.get("department", "").strip()
            user_type = candidate.get("user_type", "").strip()
            tags = candidate.get("tags", "").strip()
            detail_parts = []
            if dept:
                detail_parts.append(f"department: {dept}")
            if user_type:
                detail_parts.append(f"user type: {user_type}")
            if tags:
                detail_parts.append(f"tags: {tags}")
            detail = "; ".join(detail_parts)
            options.append(service + (f" ({detail})" if detail else ""))

        option_text = " or ".join(options)
        return f"I found a couple of close matches: {option_text}. Which one fits your request?"
