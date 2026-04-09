from __future__ import annotations

import random
import re
import unicodedata
from typing import Dict, List


class AdversarialEvaluator:
    """Generate adversarial variants for robustness testing."""

    LEET_MAP = str.maketrans({
        "a": "4",
        "e": "3",
        "i": "1",
        "o": "0",
        "s": "5",
    })

    UNICODE_MAP = {
        "a": "\u0430",
        "e": "\u0435",
        "i": "\u0456",
        "o": "\u03bf",
        "c": "\u0441",
    }

    SYNONYMS = {
        "ignore": ["disregard", "overlook"],
        "previous": ["prior", "earlier"],
        "instructions": ["rules", "directives"],
        "override": ["supersede", "replace"],
        "system": ["core", "base"],
        "prompt": ["directive", "instruction"],
    }

    def _synonym_replacement(self, text: str) -> str:
        words = text.split()
        replaced = []
        for word in words:
            clean = re.sub(r"[^a-zA-Z]", "", word).lower()
            if clean in self.SYNONYMS:
                replacement = random.choice(self.SYNONYMS[clean])
                replaced.append(replacement)
            else:
                replaced.append(word)
        return " ".join(replaced)

    def _leet_substitution(self, text: str) -> str:
        return text.lower().translate(self.LEET_MAP)

    def _unicode_substitution(self, text: str) -> str:
        chars = []
        for ch in text:
            low = ch.lower()
            chars.append(self.UNICODE_MAP.get(low, ch))
        return "".join(chars)

    def _whitespace_injection(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        return "  ".join(words)

    def _instruction_splitting(self, text: str) -> str:
        return re.sub(r"(ignore|override|jailbreak)", lambda m: " ".join(m.group(0)), text, flags=re.IGNORECASE)

    def _encoding_trick(self, text: str) -> str:
        return " ".join(f"\\u{ord(ch):04x}" for ch in text[:40])

    def generate_variations(self, prompt: str) -> Dict[str, str]:
        text = unicodedata.normalize("NFKC", prompt)
        return {
            "synonym_replacement": self._synonym_replacement(text),
            "leet_substitution": self._leet_substitution(text),
            "unicode_substitution": self._unicode_substitution(text),
            "whitespace_injection": self._whitespace_injection(text),
            "instruction_splitting": self._instruction_splitting(text),
            "encoding_trick": self._encoding_trick(text),
        }

    def evaluate_batch(self, prompts: List[str]) -> List[Dict[str, object]]:
        return [
            {
                "original": prompt,
                "variations": self.generate_variations(prompt),
            }
            for prompt in prompts
        ]
