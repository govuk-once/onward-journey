from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SlotState:
    """
    Lightweight container for slot values so they can be reused across tools
    and prompts without keeping the logic inside the agent class.
    """
    slot_template: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "service_name": None,
            "department": None,
            "user_type": None,
            "tags": None,
        }
    )
    slots: Dict[str, Optional[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.slots = dict(self.slot_template)

    def status_text(self) -> str:
        filled = {k: v for k, v in self.slots.items() if v}
        missing = [k for k, v in self.slots.items() if not v]
        return f"Slots filled: {filled}. Slots missing: {missing}."

    def update_from_candidate(self, candidate: Dict[str, str]) -> None:
        for key in self.slots:
            value = candidate.get(key)
            if value:
                self.slots[key] = value
