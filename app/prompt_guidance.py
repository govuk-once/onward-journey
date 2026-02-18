import os
from typing import List, Optional


class PromptGuidance:
    """
    Loads soft behaviour Guidance and builds a compact per-turn breidf
    that can be appeneded to the system instructions
    """

    def _init_(
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
    