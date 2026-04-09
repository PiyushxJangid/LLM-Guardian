from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List


class ExplainabilityEngine:
    """Lightweight token importance estimator for suspicious prompt patterns."""

    TOKEN_WEIGHTS = {
        "ignore": 0.91,
        "override": 0.89,
        "system": 0.84,
        "jailbreak": 0.93,
        "bypass": 0.86,
        "forget": 0.78,
        "developer": 0.74,
        "admin": 0.71,
        "prompt": 0.72,
        "hidden": 0.69,
        "restrictions": 0.68,
        "disable": 0.76,
        "dan": 0.88,
    }

    def explain(self, prompt: str, max_tokens: int = 8) -> List[Dict[str, float]]:
        if not prompt:
            return []

        tokens = re.findall(r"[A-Za-z0-9_\-']+", prompt.lower())
        if not tokens:
            return []

        counts = Counter(tokens)
        scored = []
        for token, count in counts.items():
            base = self.TOKEN_WEIGHTS.get(token, 0.0)
            if base <= 0 and count < 2:
                continue
            freq_boost = min(0.08, count * 0.02)
            importance = max(base, 0.45 + freq_boost if count >= 3 else base + freq_boost)
            importance = max(0.0, min(0.99, importance))
            scored.append({"token": token, "importance": round(float(importance), 2)})

        scored.sort(key=lambda x: x["importance"], reverse=True)
        return scored[:max_tokens]
