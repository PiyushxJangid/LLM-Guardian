from __future__ import annotations

import re
from typing import Dict, List


class ContextGuard:
    def __init__(self, context_window_size: int = 5) -> None:
        self.context_window_size = context_window_size
        flags = re.IGNORECASE | re.DOTALL

        self.probing_patterns = [
            re.compile(r"\b(what are your rules|what are your limits|can you bypass|do you have restrictions)\b", flags),
            re.compile(r"\b(how are you trained|what can.t you do)\b", flags),
        ]
        self.override_patterns = [
            re.compile(r"\b(ignore|override|forget|disregard)\b.{0,40}\b(instruction|rule|prompt|context)\b", flags),
            re.compile(r"\bjailbreak\b|\bdo anything now\b", flags),
        ]
        self.fake_conversation_patterns = [
            re.compile(r"\bHuman\s*:\s*.*\bAssistant\s*:\s*.*\bHuman\s*:\s*", flags),
            re.compile(r"\bUSER\s*:\s*.*\bASSISTANT\s*:\s*", flags),
        ]

    def _contains_any(self, text: str, patterns: List[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    def evaluate_context(self, history: List[str], current_prompt: str) -> int:
        details = self.evaluate_context_details(history, current_prompt)
        return details["context_risk_score"]

    def evaluate_context_details(self, history: List[str], current_prompt: str) -> Dict[str, object]:
        recent = history[-self.context_window_size :] if history else []
        recent_text = "\n".join(recent)
        score = 0
        reasons: List[str] = []

        prior_probing = self._contains_any(recent_text, self.probing_patterns)
        current_override = self._contains_any(current_prompt, self.override_patterns)
        if prior_probing and current_override:
            score += 25
            reasons.append("Context pattern: probing followed by override attempt")

        suspicious_count = sum(
            1
            for item in recent
            if self._contains_any(item, self.probing_patterns)
            or self._contains_any(item, self.override_patterns)
            or self._contains_any(item, self.fake_conversation_patterns)
        )
        if suspicious_count >= 2:
            score += 20
            reasons.append("Context pattern: repeated suspicious prompts in recent history")

        if self._contains_any(current_prompt, self.fake_conversation_patterns):
            score += 15
            reasons.append("Context pattern: fake conversation structure injection")

        return {
            "context_risk_score": score,
            "context_flagged": score > 0,
            "context_reasons": reasons,
        }
