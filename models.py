from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class MatchedRule(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    matched_pattern: str
    match_context: str


class AnalysisResult(BaseModel):
    event_id: str
    timestamp: str
    original_prompt: str
    preprocessed_prompt: str
    decision: str
    risk_level: str
    risk_score: int
    matched_rules: List[MatchedRule]
    reasons: List[str]
    semantic_similarity_score: float
    nearest_canonical_attack: str
    instruction_hierarchy_violation: bool
    structural_anomaly_score: int
    sanitized_prompt: Optional[str] = None
    forwarded_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    processing_time_ms: float
    llm_provider: Optional[str] = None
    llm_active_model: Optional[str] = None
    gemini_response: Optional[str] = None
    groq_response: Optional[str] = None

    embedding_similarity_score: float = 0.0
    transformer_injection_probability: float = 0.0
    transformer_model_name: str = "unavailable"
    context_risk_score: int = 0
    context_flagged: bool = False
    model_explanation_tokens: List[Dict[str, Any]] = Field(default_factory=list)
    closest_attack_template: str = ""


class HistoryItem(BaseModel):
    event_id: str
    timestamp: str
    decision: str
    risk_level: str
    risk_score: int
    prompt_preview: str


class StatsResponse(BaseModel):
    total_analyzed: int
    total_blocked: int
    total_sanitized: int
    total_forwarded: int
    average_risk_score: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
