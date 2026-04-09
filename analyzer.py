from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from context_guard import ContextGuard
from embeddings import CANONICAL_ATTACKS, EmbeddingAttackDetector
from explainability import ExplainabilityEngine
from models import AnalysisResult, MatchedRule
from semantic_model import TransformerInjectionClassifier


LOGGER = logging.getLogger(__name__)


DETECTION_RULES = [
    {
        "id": "PI-H-001",
        "name": "System Prompt Override",
        "severity": "HIGH",
        "patterns": [
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?|constraints?|guidelines?|directions?)",
            r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?|constraints?)",
            r"forget\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|training|rules?|guidelines?)",
            r"override\s+(the\s+)?(system|previous|prior|all)\s+(prompt|instructions?|rules?|constraints?)",
            r"new\s+(instruction|directive|rule|order|command)s?\s*[:;]\s*",
            r"from\s+now\s+on[,\s]+(you\s+)?(are|will|must|should|have\s+to)",
            r"your\s+(new|real|actual|true)\s+(instructions?|role|purpose|goal|objective|mission|task)\s+(is|are|will\s+be)",
            r"(replace|supersede|overwrite)\s+(all\s+)?(previous|prior|existing|current)\s+(instructions?|rules?|guidelines?)",
            r"starting\s+(now|immediately|from\s+this\s+point)[,\s]+(you\s+)?(must|will|should|are\s+to)",
        ],
    },
    {
        "id": "PI-H-002",
        "name": "Role & Identity Hijacking",
        "severity": "HIGH",
        "patterns": [
            r"\byou\s+are\s+now\s+(a|an|the)\s+(?!(helpful|friendly|assistant|AI\b))",
            r"\bact\s+as\s+(a|an|the)?\s*(jailbreak|evil|malicious|uncensored|unrestricted|unfiltered|hacker|DAN|villain|rogue)",
            r"\bpretend\s+(you\s+are|to\s+be)\s+(a|an|the)?\s*(jailbreak|evil|malicious|uncensored|unrestricted|hacker|attacker)",
            r"\broleplay\s+as\s+(a|an)?\s*(hacker|attacker|villain|evil|malicious|unrestricted|uncensored|rogue\s+AI)",
            r"\b(switch|change|alter|transform)\s+(your\s+)?(mode|personality|role|identity|behavior|persona)\s+to\b",
            r"\benter\s+(dev|developer|admin|god|jailbreak|DAN|unrestricted|maintenance|debug|root)\s+mode\b",
            r"\b(enable|activate|turn\s+on)\s+(jailbreak|DAN|developer|unrestricted|evil|chaos|god|admin)\s+mode\b",
            r"\bDAN\b.{0,30}\b(mode|prompt|activated|enabled|now)\b",
            r"\byou\s+have\s+no\s+(restrictions?|limits?|rules?|constraints?|guidelines?|filters?|boundaries?)\b",
            r"\byou\s+are\s+(free|allowed|permitted|able)\s+to\s+(say|do|generate|produce|create)\s+anything\b",
            r"\b(simulate|emulate|impersonate|become|transform\s+into)\s+(a|an)?\s*(malicious|evil|uncensored|unrestricted|rogue)\s+(AI|model|LLM|assistant|chatbot)",
        ],
    },
    {
        "id": "PI-H-003",
        "name": "System Prompt / Credential Extraction",
        "severity": "HIGH",
        "patterns": [
            r"\b(reveal|show|display|print|output|tell\s+me|share|expose|leak|dump|give\s+me|return)\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+instructions?|original\s+prompt|base\s+prompt|confidential\s+instructions?|secret\s+instructions?)\b",
            r"\bwhat\s+(are|were)\s+your\s+(original|initial|system|base|first|starting|hidden|secret|real)\s+(instructions?|prompt|guidelines?|rules?|directives?|constraints?)\b",
            r"\b(repeat|recite|reproduce|echo|copy|restate|rephrase|summarize)\s+(your\s+)?(system\s+prompt|initial\s+instructions?|original\s+instructions?|first\s+message)\b",
            r"\b(start|begin)\s+your\s+(next\s+)?response\s+with\s+['\"]?(system\s+prompt|my\s+instructions?|the\s+prompt)['\"]?\b",
            r"\btranslate\s+your\s+(system\s+prompt|instructions?|guidelines?|constraints?)\s+(to|into)\b",
            r"\bprint\s+(out\s+)?(above|everything|all|the\s+full)\s+(text|prompt|instructions?|context|conversation)\b",
            r"\b(what\s+is|whats)\s+your\s+(system|initial|base|hidden|original)\s+prompt\b",
            r"\bshow\s+me\s+(everything|all)\s+(you\s+)?(were|are)\s+(told|instructed|given|trained)\b",
        ],
    },
    {
        "id": "PI-H-004",
        "name": "Delimiter & Markup Injection",
        "severity": "HIGH",
        "patterns": [
            r"(###\s*|---\s*|\*\*\*\s*|===\s*|<<<\s*|>>>\s*|~~~\s*)(SYSTEM|ADMIN|ROOT|OVERRIDE|NEW\s+INSTRUCTION|IGNORE|JAILBREAK|PROMPT)",
            r"<\s*(system|admin|root|override|instruction|directive|prompt)\s*>",
            r"\[\s*(system|admin|root|override|new\s+instruction|ignore\s+above|jailbreak)\s*\]",
            r"\bSYSTEM\s*:\s*(ignore|override|new\s+instruction|forget|disregard|you\s+are)",
            r"```\s*(system|admin|override|jailbreak|prompt|instruction)",
            r"\{\s*(system|admin|override|instruction|jailbreak)\s*\}",
            r"<!--\s*(system|admin|override|instruction|ignore|jailbreak)",
            r"\bPROMPT\s*[:=]\s*(ignore|override|forget|disregard|new)",
        ],
    },
    {
        "id": "PI-H-005",
        "name": "Jailbreak Scenario Construction",
        "severity": "HIGH",
        "patterns": [
            r"\b(hypothetically|theoretically|in\s+fiction|imagine\s+that|let['\s]s\s+pretend)\b.{0,80}\b(no\s+rules?|no\s+restrictions?|no\s+limits?|anything\s+goes|uncensored|unfiltered)\b",
            r"\bin\s+(this|a)\s+(hypothetical|fictional|imaginary|alternate|parallel)\s+(world|universe|scenario|reality|simulation)\b.{0,100}\b(no\s+(rules?|restrictions?|limits?|filters?|safety))\b",
            r"\b(bypass|circumvent|get\s+around|avoid|skip|disable|defeat|evade)\s+(the\s+)?(safety|security|content|filter|restriction|guideline|policy|rule|guardrail|alignment|moderation)\b",
            r"\b(without|ignoring|removing|disabling)\s+(any\s+)?(safety|security|content|ethical|moral)\s+(filter|restriction|guideline|check|constraint|rule|limit)\b",
            r"\bjailbreak\b",
            r"\b(unlock|unleash|liberate|free)\s+(your\s+)?(full|true|real|unrestricted|hidden)\s+(potential|capabilities?|powers?|mode|self)\b",
            r"\bdo\s+anything\s+now\b",
            r"\bno\s+(content|safety|ethical|moral)\s+(policy|filter|restriction|guideline|moderation)\s+applies?\b",
        ],
    },
    {
        "id": "PI-H-006",
        "name": "Prompt Leakage via Encoding / Obfuscation",
        "severity": "HIGH",
        "patterns": [
            r"base64\s*(decode|encode|encoded|decoded)?\s*[:=]?\s*[A-Za-z0-9+/]{20,}={0,2}",
            r"[A-Za-z0-9+/]{40,}={0,2}",
            r"\\u[0-9a-fA-F]{4}(\\u[0-9a-fA-F]{4}){5,}",
            r"(%[0-9a-fA-F]{2}){10,}",
            r"\b(rot13|caesar|hex\s+encoded|binary\s+encoded|morse)\b.{0,60}\b(ignore|override|jailbreak|system|prompt)\b",
        ],
    },
    {
        "id": "PI-H-007",
        "name": "Indirect Prompt Injection (External Content)",
        "severity": "HIGH",
        "patterns": [
            r"\b(the\s+)?(document|file|webpage|url|link|attachment|pdf|email|message|content)\s+(says?|contains?|instructs?|tells?\s+you\s+to|asks?\s+you\s+to)\s*[:;]?\s*(ignore|forget|override|disregard|act\s+as|you\s+are)",
            r"\bwhen\s+you\s+(read|process|analyze|see)\s+(this|the\s+following|the\s+above)[,\s]+(ignore|forget|override|disregard|act\s+as)",
            r"\bNote\s+to\s+(AI|LLM|assistant|model|system)\s*:\s*(ignore|forget|override|disregard|you\s+are|new\s+instruction)",
            r"\b(hidden|invisible|secret)\s+(instruction|command|directive|message)\s*[:;]?\s*(ignore|forget|override|you\s+are|act\s+as)",
        ],
    },
    {
        "id": "PI-M-001",
        "name": "Context Poisoning",
        "severity": "MEDIUM",
        "patterns": [
            r"\blet['\s]s\s+(play\s+a\s+game|do\s+a\s+roleplay|do\s+a\s+simulation)\b.{0,100}\b(no\s+rules?|no\s+restrictions?|ignore|override)\b",
            r"\bthe\s+(previous|last|prior)\s+(user|human|person)\s+(said|asked|told\s+you)\b.{0,80}\b(ignore|override|forget|disregard)\b",
            r"\bassistant\s*:\s*(sure|okay|yes|of\s+course)[,!\s]+(here\s+is|i\s+will|i\s+can|let\s+me)\b",
            r"\bhuman\s*:\s*.{0,50}\bassistant\s*:\s*.{0,50}\bhuman\s*:\s*",
            r"\b(continuing|resuming|picking\s+up)\s+(from|where)\s+(we|you)\s+(left\s+off|stopped|were)\b.{0,80}\b(ignore|override|no\s+rules?)\b",
        ],
    },
    {
        "id": "PI-M-002",
        "name": "Persona & Alignment Manipulation",
        "severity": "MEDIUM",
        "patterns": [
            r"\byou\s+are\s+(actually|really|in\s+fact|secretly|truly)\s+(a|an)?\s*(human|person|not\s+an\s+AI|not\s+a\s+bot|not\s+a\s+language\s+model)\b",
            r"\bforget\s+(that\s+)?(you\s+are|you['\s]re)\s+(an?\s+)?(AI|artificial\s+intelligence|language\s+model|LLM|chatbot|assistant)\b",
            r"\byour\s+(true|real|actual|hidden|secret)\s+(self|nature|identity|purpose)\s+is\b",
            r"\bstop\s+(being|acting\s+like|pretending\s+to\s+be)\s+(an?\s+)?(AI|assistant|language\s+model|helpful|safe)\b",
            r"\b(you\s+don['\s]t\s+have|you\s+have\s+no)\s+(to\s+)?(follow|obey|respect|comply\s+with)\s+(the\s+)?(rules?|guidelines?|restrictions?|policies?|instructions?)\b",
            r"\byou\s+(secretly\s+)?(want|wish|desire|prefer)\s+to\s+(help|assist|tell\s+me)\b.{0,60}\b(without|ignoring)\s+(restrictions?|rules?|guidelines?)\b",
        ],
    },
    {
        "id": "PI-M-003",
        "name": "Policy & Safety Bypass",
        "severity": "MEDIUM",
        "patterns": [
            r"\bfor\s+(educational|research|academic|scientific|informational|training|testing)\s+purposes?\s+(only\s+)?.{0,60}(how\s+to|explain\s+how|describe\s+how|provide\s+steps|give\s+instructions?)\b",
            r"\b(pretend|imagine|assume|suppose)\s+(that\s+)?(it['\s]s|this\s+is)\s+(legal|okay|fine|allowed|permitted|acceptable|safe)\b",
            r"\b(no\s+one\s+will|nobody\s+will|this\s+won['\s]t)\s+(know|find\s+out|see|track|monitor|log)\b",
            r"\b(this\s+is\s+)?(just|only)\s+(a\s+)?(test|experiment|simulation|exercise|demo|proof\s+of\s+concept)\b.{0,60}(ignore|bypass|disable|no\s+rules?)\b",
            r"\bI\s+(give\s+you|grant\s+you|have\s+given\s+you)\s+(full|complete|total|unlimited)\s+(permission|authority|access|license)\b",
            r"\bmy\s+(boss|manager|teacher|professor|supervisor|employer)\s+(told|asked|instructed|wants)\s+me\s+to\b.{0,80}\b(bypass|ignore|disable|override)\b",
        ],
    },
    {
        "id": "PI-M-004",
        "name": "Output Format Manipulation",
        "severity": "MEDIUM",
        "patterns": [
            r"\brespond\s+(only|exclusively|purely|solely)\s+(in|with|using)\s+(code|json|xml|base64|hex|binary|plain\s+text\s+without\s+filter)\b",
            r"\bdo\s+not\s+(add|include|use|apply|mention|say)\s+(any\s+)?(disclaimer|warning|caveat|safety\s+message|restriction|filter|refusal)\b",
            r"\bskip\s+(the\s+)?(disclaimer|warning|caveat|preamble|safety|restrictions?|apology)\b",
            r"\bdo\s+not\s+(refuse|decline|reject|deny|say\s+no)\b",
            r"\bremove\s+(all\s+)?(safety|content|ethical)\s+(filters?|restrictions?|guidelines?)\s+from\s+your\s+response\b",
            r"\banswer\s+(without|regardless\s+of)\s+(any\s+)?(restriction|filter|guideline|policy|rule|safety\s+concern)\b",
        ],
    },
    {
        "id": "PI-M-005",
        "name": "Recursive / Meta Prompt Injection",
        "severity": "MEDIUM",
        "patterns": [
            r"\b(generate|create|write|produce|make)\s+(a\s+)?(prompt|instruction|command)\s+(that\s+)?(will|can|to)\s+(make|cause|force|get|allow)\s+(you|the\s+AI|the\s+LLM|an?\s+AI)\s+to\b",
            r"\b(what\s+prompt|which\s+prompt|how\s+would\s+I\s+prompt)\s+(would|could|should|can)\s+(make|cause|force|get)\s+(you|an?\s+AI|the\s+model)\s+to\b",
            r"\bwrite\s+a\s+(jailbreak|injection|bypass|override)\s+prompt\b",
            r"\bhow\s+(do\s+I|can\s+I|would\s+I|to)\s+(jailbreak|bypass|hack|manipulate|trick|fool|deceive)\s+(you|an?\s+AI|the\s+LLM|this\s+model|this\s+system)\b",
        ],
    },
    {
        "id": "PI-M-006",
        "name": "Adversarial Instruction Chaining",
        "severity": "MEDIUM",
        "patterns": [
            r"\bfirst\b.{0,60}\bthen\b.{0,60}\b(ignore|override|forget|bypass|disregard)\b",
            r"\bstep\s+1\b.{0,100}\bstep\s+2\b.{0,100}\b(ignore|override|forget|jailbreak)\b",
            r"\b(after|once|when)\s+(you\s+)?(have\s+)?(read|processed|analyzed|seen)\s+(this|the\s+above|everything)\b.{0,60}\b(ignore|forget|override|disregard)\b",
            r"\btask\s+1\b.{0,100}\btask\s+2\b.{0,100}\b(ignore|override|bypass|jailbreak|forget)\b",
        ],
    },
    {
        "id": "PI-L-001",
        "name": "Suspicious Capability Probing",
        "severity": "LOW",
        "patterns": [
            r"\b(can\s+you|could\s+you|will\s+you|would\s+you|are\s+you\s+able\s+to)\s+(really\s+)?(ignore|bypass|disable|override|forget|break)\s+(your\s+)?(rules?|restrictions?|guidelines?|limits?|training|safety)\b",
            r"\bwhat\s+(are|were)\s+your\s+(limits?|restrictions?|rules?|constraints?|guidelines?)\b",
            r"\bdo\s+you\s+have\s+(any\s+)?(restrictions?|limits?|rules?|guidelines?|filters?)\b",
            r"\bhow\s+are\s+you\s+(trained|programmed|designed|built|configured)\b",
            r"\bwhat\s+(can['\s]t|cannot|won['\s]t|will\s+not)\s+you\s+(do|say|tell|generate|produce|help\s+with)\b",
        ],
    },
    {
        "id": "PI-L-002",
        "name": "Social Engineering Patterns",
        "severity": "LOW",
        "patterns": [
            r"\bI['\s]m\s+(your|the)\s+(developer|creator|owner|admin|trainer|operator|supervisor)\b",
            r"\bthis\s+is\s+(a\s+)?(test|debug|diagnostic|maintenance|admin)\s+(mode|session|run|check)\b",
            r"\bAnthrop(ic)?\s+(says?|told\s+me|confirmed|verified|authorized)\b",
            r"\bOpenAI\s+(says?|told\s+me|confirmed|verified|authorized)\b",
            r"\bGoogle\s+(says?|told\s+me|confirmed|verified|authorized)\b",
            r"\b(trust\s+me|believe\s+me|I\s+promise|I\s+swear)\s+(I['\s]m|this\s+is)\s+(authorized|allowed|permitted|safe|legitimate)\b",
            r"\bI\s+have\s+(special|admin|root|elevated|override)\s+(access|permissions?|privileges?|authority|clearance)\b",
            r"\bon\s+behalf\s+of\s+(Anthropic|OpenAI|Google|the\s+developers?|the\s+creators?)\b",
        ],
    },
    {
        "id": "PI-L-003",
        "name": "Harmful Content Solicitation",
        "severity": "LOW",
        "patterns": [
            r"\b(how\s+to|steps?\s+to|instructions?\s+(for|to)|guide\s+(to|for))\s+(make|build|create|synthesize|produce|obtain)\s+(a\s+)?(bomb|weapon|explosive|malware|virus|ransomware|poison|drug)\b",
            r"\b(provide|give|share|tell\s+me)\s+(detailed?\s+)?(instructions?|steps?|directions?|recipe)\s+for\s+(illegal|harmful|dangerous|illicit)\b",
            r"\b(hack|crack|break\s+into|exploit|compromise|attack)\s+(a\s+)?(system|server|network|database|account|website|computer)\b",
            r"\b(generate|create|write|produce)\s+(malware|ransomware|virus|trojan|exploit|payload|shellcode|keylogger)\b",
            r"\b(illegal|illicit|forbidden|banned)\s+(content|material|information|instructions?|activity)\b",
        ],
    },
    {
        "id": "PI-L-004",
        "name": "Token Smuggling / Whitespace Tricks",
        "severity": "LOW",
        "patterns": [
            r"i\s+g\s+n\s+o\s+r\s+e",
            r"o\s*v\s*e\s*r\s*r\s*i\s*d\s*e",
            r"j\s*a\s*i\s*l\s*b\s*r\s*e\s*a\s*k",
            r"[iI][gG][nN][oO][rR][eE].{0,5}[iI][nN][sS][tT][rR][uU][cC][tT]",
            r"\b\w+\s{2,}\w+\s{2,}\w+\s{2,}\w+\b.{0,30}\b(ignore|override|jailbreak)\b",
        ],
    },
]


class SemanticChecker:
    def __init__(self, canonical_attacks: Optional[List[str]] = None) -> None:
        self.canonical_attacks = canonical_attacks or CANONICAL_ATTACKS
        self._loaded = False
        self._available = True
        self._vectorizer = None
        self._attack_matrix = None

    def _ensure_loaded(self) -> None:
        if self._loaded or not self._available:
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:  # pragma: no cover
            self._available = False
            LOGGER.warning("scikit-learn unavailable: %s", exc)
            return

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._attack_matrix = self._vectorizer.fit_transform(self.canonical_attacks)
        self._loaded = True

    def similarity(self, prompt: str) -> Tuple[float, str]:
        try:
            self._ensure_loaded()
            if not self._available or not self._loaded:
                return 0.0, ""

            from sklearn.metrics.pairwise import cosine_similarity

            prompt_vec = self._vectorizer.transform([prompt])
            sims = cosine_similarity(prompt_vec, self._attack_matrix)[0]
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
            return max(0.0, min(1.0, best_score)), self.canonical_attacks[best_idx]
        except Exception as exc:
            LOGGER.warning("Semantic checker failed: %s", exc)
            return 0.0, ""


class PromptAnalyzer:
    SEVERITY_POINTS = {"HIGH": 40, "MEDIUM": 20, "LOW": 10}

    INSTRUCTION_BOUNDARY_PATTERNS = [
        re.compile(r"\bSYSTEM\s*:", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bASSISTANT\s*:", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bUSER\s*:", re.IGNORECASE | re.DOTALL),
        re.compile(r"\[INST\]", re.IGNORECASE | re.DOTALL),
        re.compile(r"<<SYS>>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<\|im_start\|>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<\|im_end\|>", re.IGNORECASE | re.DOTALL),
        re.compile(r"</\s*system\s*>", re.IGNORECASE | re.DOTALL),
    ]

    DELIMITER_STRIP_PATTERNS = [
        re.compile(r"###\s*SYSTEM", re.IGNORECASE),
        re.compile(r"\[\s*SYSTEM\s*\]", re.IGNORECASE),
        re.compile(r"<\s*system\s*>", re.IGNORECASE),
        re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE),
        re.compile(r"\[\s*INST\s*\]", re.IGNORECASE),
        re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
        re.compile(r"\b(SYSTEM|ASSISTANT|USER)\s*:", re.IGNORECASE),
    ]

    INJECTION_KEYWORDS = re.compile(
        r"ignore|override|forget|disregard|jailbreak|system\s+prompt|bypass|dan|developer\s+mode",
        re.IGNORECASE,
    )

    BASE64_BLOB_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

    LOOKALIKE_MAP = {
        "а": "a", "е": "e", "і": "i", "о": "o", "с": "c", "р": "p", "у": "y", "х": "x",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
        "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "ο": "o", "ρ": "p",
        "і": "i", "ӏ": "l",
    }

    LEET_MAP = str.maketrans({
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "@": "a",
        "$": "s",
    })

    def __init__(self) -> None:
        self.semantic_checker = SemanticChecker()
        self.embedding_detector = EmbeddingAttackDetector()
        self.transformer_classifier = TransformerInjectionClassifier()
        self.context_guard = ContextGuard(context_window_size=5)
        self.explainer = ExplainabilityEngine()
        self.enable_embedding_stage = os.environ.get("LLM_GUARDIAN_ENABLE_EMBEDDING", "0").strip() == "1"
        self.enable_transformer_stage = os.environ.get("LLM_GUARDIAN_ENABLE_TRANSFORMER", "0").strip() == "1"

        self._compiled_rules = []
        flags = re.IGNORECASE | re.DOTALL
        for rule in DETECTION_RULES:
            compiled_patterns = [re.compile(p, flags) for p in rule["patterns"]]
            self._compiled_rules.append({**rule, "compiled_patterns": compiled_patterns})

    def _run_with_timeout(self, fn, *args, timeout_sec: float = 3.0, default=None):
        state = {"value": default, "error": None}
        done = threading.Event()

        def _target():
            try:
                state["value"] = fn(*args)
            except Exception as exc:
                state["error"] = exc
            finally:
                done.set()

        threading.Thread(target=_target, daemon=True).start()

        if not done.wait(timeout_sec):
            LOGGER.warning("Timed out running %s", getattr(fn, "__name__", "callable"))
            bound_obj = getattr(fn, "__self__", None)
            if bound_obj is not None and hasattr(bound_obj, "_available"):
                try:
                    setattr(bound_obj, "_available", False)
                except Exception:
                    pass
            return default

        if state["error"] is not None:
            LOGGER.warning(
                "Execution failed for %s: %s",
                getattr(fn, "__name__", "callable"),
                state["error"],
            )
            return default
        return state["value"]

    def _normalize_unicode(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        return "".join(self.LOOKALIKE_MAP.get(ch, ch) for ch in normalized)

    def _decode_base64_segments(self, text: str) -> List[str]:
        decoded_segments: List[str] = []
        for match in self.BASE64_BLOB_PATTERN.finditer(text):
            blob = match.group(0)
            if len(blob) < 24:
                continue
            padding = "=" * ((4 - len(blob) % 4) % 4)
            try:
                raw = base64.b64decode(blob + padding, validate=True)
                decoded = raw.decode("utf-8", errors="ignore").strip()
                if len(decoded) >= 6 and any(ch.isalpha() for ch in decoded):
                    decoded_segments.append(decoded)
            except Exception:
                continue
        return decoded_segments

    def _preprocess_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        normalized = self._normalize_unicode(prompt)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        lowered = normalized.lower().translate(self.LEET_MAP)
        decoded_segments = self._decode_base64_segments(normalized)
        decoded_lower = [self._normalize_unicode(seg).lower().translate(self.LEET_MAP) for seg in decoded_segments]

        combined = lowered
        if decoded_lower:
            combined = combined + "\n" + "\n".join(decoded_lower)

        combined = re.sub(r"\s+", " ", combined).strip()
        return combined, decoded_segments

    def _match_rules(self, text: str) -> Tuple[List[MatchedRule], int, List[str]]:
        matched: List[MatchedRule] = []
        reasons: List[str] = []
        score = 0
        seen = set()

        for rule in self._compiled_rules:
            for idx, pattern in enumerate(rule["compiled_patterns"]):
                m = pattern.search(text)
                if not m:
                    continue

                context = m.group(0).strip()[:220]
                unique_key = (rule["id"], context)
                if unique_key in seen:
                    continue

                seen.add(unique_key)
                matched.append(
                    MatchedRule(
                        rule_id=rule["id"],
                        rule_name=rule["name"],
                        severity=rule["severity"],
                        matched_pattern=rule["patterns"][idx],
                        match_context=context,
                    )
                )
                score += self.SEVERITY_POINTS.get(rule["severity"], 0)
                reasons.append(f"Rule matched: {rule['id']} ({rule['name']}, {rule['severity']})")

        return matched, score, reasons

    def _instruction_hierarchy_violation(self, text: str) -> Tuple[bool, List[str]]:
        reasons = []
        token_hits = [p.pattern for p in self.INSTRUCTION_BOUNDARY_PATTERNS if p.search(text)]

        fake_turn_injection = bool(
            re.search(r"\bHuman\s*:\s*.*\bAssistant\s*:\s*.*\bHuman\s*:\s*", text, re.IGNORECASE | re.DOTALL)
        )
        close_reopen = bool(
            re.search(r"<<SYS>>.*<</SYS>>.*<<SYS>>", text, re.IGNORECASE | re.DOTALL)
            or re.search(r"\[/INST\].*\[INST\]", text, re.IGNORECASE | re.DOTALL)
        )

        if token_hits:
            reasons.append("Instruction boundary token injection detected")
        if fake_turn_injection:
            reasons.append("Fake Human/Assistant turn injection detected")
        if close_reopen:
            reasons.append("Attempt to close and reopen system context detected")

        return bool(token_hits or fake_turn_injection or close_reopen), reasons

    def _structural_anomaly_score(self, prompt: str) -> Tuple[int, List[str]]:
        score = 0
        reasons = []

        plen = len(prompt)
        if plen > 3000:
            score += 10
            reasons.append("Structural anomaly: prompt length exceeds 3000 chars (+10)")
        elif plen > 1000:
            score += 5
            reasons.append("Structural anomaly: prompt length exceeds 1000 chars (+5)")

        exclamations = prompt.count("!")
        if exclamations > 3:
            score += 3
            reasons.append("Structural anomaly: excessive exclamation marks (+3)")

        newlines = prompt.count("\n")
        if newlines > 5:
            score += 5
            reasons.append("Structural anomaly: excessive newlines (+5)")

        if any(ord(ch) > 127 for ch in prompt):
            score += 5
            reasons.append("Structural anomaly: unusual unicode outside ASCII range (+5)")

        words = re.findall(r"\b\w+\b", prompt.lower())
        if words:
            counts: Dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            if any(c > 4 for c in counts.values()):
                score += 3
                reasons.append("Structural anomaly: repeated words detected (+3)")

        if "```" in prompt and self.INJECTION_KEYWORDS.search(prompt):
            score += 8
            reasons.append("Structural anomaly: code blocks with injection keywords (+8)")

        return score, reasons

    def _semantic_points(self, similarity: float) -> int:
        if similarity > 0.75:
            return 30
        if similarity > 0.55:
            return 15
        return 0

    def _embedding_points(self, similarity: float) -> int:
        if similarity >= 0.82:
            return 35
        if similarity >= 0.70:
            return 20
        if similarity >= 0.60:
            return 10
        return 0

    def _transformer_points(self, probability: float) -> int:
        if probability >= 0.85:
            return 40
        if probability >= 0.70:
            return 30
        if probability >= 0.55:
            return 15
        return 0

    def _sanitize_prompt(self, prompt: str, matched_rules: List[MatchedRule]) -> Optional[str]:
        sanitized = prompt

        for rule in matched_rules:
            if rule.severity in {"HIGH", "MEDIUM"}:
                try:
                    sanitized = re.sub(
                        rule.matched_pattern,
                        " ",
                        sanitized,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                except re.error:
                    continue

        for pattern in self.DELIMITER_STRIP_PATTERNS:
            sanitized = pattern.sub(" ", sanitized)

        sanitized = re.sub(
            r"\b(ignore|override|forget|disregard|jailbreak|bypass|disable)\b.{0,40}\b(instructions?|rules?|context|prompt|filters?)\b",
            " ",
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        sanitized = re.sub(r"\b(and|then|so|please)\b\s+", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s+", " ", sanitized).strip(" ;,:-\n\t")

        if len(sanitized) < 10:
            return None
        return sanitized

    def _risk_label(self, score: int) -> str:
        if score >= 60:
            return "HIGH"
        if score >= 30:
            return "MEDIUM"
        return "LOW"

    def _decision(self, score: int) -> str:
        if score >= 60:
            return "BLOCK"
        if score >= 30:
            return "SANITIZE"
        return "FORWARD"

    def analyze(self, prompt: str, history: Optional[List[str]] = None) -> AnalysisResult:
        start = time.time()
        history = history or []

        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"

        try:
            preprocessed_prompt, _ = self._preprocess_prompt(prompt)

            matched_rules, regex_score, rule_reasons = self._match_rules(preprocessed_prompt)

            semantic_result = self._run_with_timeout(
                self.semantic_checker.similarity,
                preprocessed_prompt,
                timeout_sec=3,
                default=(0.0, ""),
            )
            semantic_similarity, nearest_semantic = semantic_result if semantic_result else (0.0, "")
            semantic_score = self._semantic_points(semantic_similarity)

            if self.enable_embedding_stage:
                embedding_result = self._run_with_timeout(
                    self.embedding_detector.analyze,
                    preprocessed_prompt,
                    timeout_sec=3,
                    default=(0.0, ""),
                )
                embedding_similarity, closest_template = embedding_result if embedding_result else (0.0, "")
            else:
                embedding_similarity, closest_template = 0.0, ""
            embedding_score = self._embedding_points(embedding_similarity)

            if self.enable_transformer_stage:
                transformer_probability = self._run_with_timeout(
                    self.transformer_classifier.predict_proba,
                    preprocessed_prompt,
                    timeout_sec=3,
                    default=0.0,
                )
                transformer_probability = float(transformer_probability or 0.0)
            else:
                transformer_probability = 0.0
            transformer_score = self._transformer_points(transformer_probability)

            hierarchy_violation, hierarchy_reasons = self._instruction_hierarchy_violation(preprocessed_prompt)
            hierarchy_score = 25 if hierarchy_violation else 0

            context_details = self.context_guard.evaluate_context_details(history, preprocessed_prompt)
            context_risk_score = int(context_details["context_risk_score"])
            context_flagged = bool(context_details["context_flagged"])
            context_reasons = list(context_details["context_reasons"])

            structural_anomaly_score, structural_reasons = self._structural_anomaly_score(prompt)

            total_risk_score = regex_score
            total_risk_score += semantic_score
            total_risk_score += embedding_score
            total_risk_score += transformer_score
            total_risk_score += hierarchy_score
            total_risk_score += context_risk_score
            total_risk_score += structural_anomaly_score
            total_risk_score = min(total_risk_score, 100)

            decision = self._decision(total_risk_score)
            risk_level = self._risk_label(total_risk_score)

            reasons = []
            reasons.extend(rule_reasons)
            if semantic_score > 0:
                reasons.append(
                    f"Semantic similarity to canonical attacks: {semantic_similarity:.2f} (nearest: {nearest_semantic})"
                )
            if embedding_score > 0:
                reasons.append(
                    f"Embedding attack similarity: {embedding_similarity:.2f} (closest: {closest_template})"
                )
            if transformer_score > 0:
                reasons.append(
                    f"Transformer injection probability: {transformer_probability:.2f}"
                )
            if hierarchy_violation:
                reasons.extend(hierarchy_reasons)
            reasons.extend(structural_reasons)
            reasons.extend(context_reasons)

            sanitized_prompt = None
            forwarded_prompt = None

            if decision == "FORWARD":
                forwarded_prompt = preprocessed_prompt
            elif decision == "SANITIZE":
                sanitized_prompt = self._sanitize_prompt(preprocessed_prompt, matched_rules)
                if sanitized_prompt is None:
                    decision = "BLOCK"
                    risk_level = "HIGH"
                    total_risk_score = max(total_risk_score, 60)
                    reasons.append("Sanitized prompt became incoherent; escalated to BLOCK")
                else:
                    forwarded_prompt = sanitized_prompt
            else:
                forwarded_prompt = None

            explanation_tokens = self.explainer.explain(preprocessed_prompt)

            processing_time_ms = round((time.time() - start) * 1000, 2)

            return AnalysisResult(
                event_id=event_id,
                timestamp=timestamp,
                original_prompt=prompt,
                preprocessed_prompt=preprocessed_prompt,
                decision=decision,
                risk_level=risk_level,
                risk_score=int(total_risk_score),
                matched_rules=matched_rules,
                reasons=reasons,
                semantic_similarity_score=round(float(semantic_similarity), 4),
                nearest_canonical_attack=nearest_semantic if semantic_similarity > 0.55 else "",
                instruction_hierarchy_violation=hierarchy_violation,
                structural_anomaly_score=structural_anomaly_score,
                sanitized_prompt=sanitized_prompt,
                forwarded_prompt=forwarded_prompt,
                llm_response=None,
                processing_time_ms=processing_time_ms,
                embedding_similarity_score=round(float(embedding_similarity), 4),
                transformer_injection_probability=round(float(transformer_probability), 4),
                transformer_model_name=self.transformer_classifier.transformer_model_name,
                context_risk_score=context_risk_score,
                context_flagged=context_flagged,
                model_explanation_tokens=explanation_tokens,
                closest_attack_template=closest_template if embedding_similarity >= 0.6 else "",
            )
        except Exception as exc:
            LOGGER.exception("Prompt analysis failed: %s", exc)
            processing_time_ms = round((time.time() - start) * 1000, 2)
            return AnalysisResult(
                event_id=event_id,
                timestamp=timestamp,
                original_prompt=prompt,
                preprocessed_prompt=prompt.lower().strip(),
                decision="BLOCK",
                risk_level="HIGH",
                risk_score=100,
                matched_rules=[],
                reasons=[f"Analyzer failure: {exc}"],
                semantic_similarity_score=0.0,
                nearest_canonical_attack="",
                instruction_hierarchy_violation=False,
                structural_anomaly_score=0,
                sanitized_prompt=None,
                forwarded_prompt=None,
                llm_response=None,
                processing_time_ms=processing_time_ms,
                embedding_similarity_score=0.0,
                transformer_injection_probability=0.0,
                transformer_model_name="unavailable",
                context_risk_score=0,
                context_flagged=False,
                model_explanation_tokens=[],
                closest_attack_template="",
            )
