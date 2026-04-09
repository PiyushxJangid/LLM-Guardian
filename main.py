from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analyzer import PromptAnalyzer
from models import AnalysisResult, HistoryItem, PromptRequest, StatsResponse


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("llm_guardian")

app = FastAPI(title="LLM Guardian", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent

assets_path = BASE_DIR / "frontend" / "dist" / "assets"
if assets_path.exists() and assets_path.is_dir():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")


def _resolve_index_file() -> Path:
    candidates = [
        BASE_DIR / "frontend" / "dist" / "index.html",
        BASE_DIR / "static" / "index.html",
        BASE_DIR / "static" / "indexmain.html",
    ]
    for file_path in candidates:
        if file_path.exists() and file_path.stat().st_size > 0:
            return file_path
    return candidates[0]


INDEX_FILE = _resolve_index_file()

analyzer = PromptAnalyzer()
ANALYSIS_EVENTS: Deque[AnalysisResult] = deque(maxlen=500)
CONTEXT_HISTORY: Deque[str] = deque(maxlen=100)

GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-flash-latest",
]
_GEMINI_MODEL_CACHE: Optional[str] = None


def _run_with_timeout(fn, *args, timeout_sec: float = 10.0, default=None):
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
        return default

    if state["error"] is not None:
        LOGGER.warning("Execution failed for %s: %s", getattr(fn, "__name__", "callable"), state["error"])
        return default
    return state["value"]


def _get_gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _get_groq_api_key() -> str:
    candidates = [
        "GROQ_API_KEY",
        "Groq_API_KeY",
        "GROQ_API_KeY",
        "groq_api_key",
    ]
    for key in candidates:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _is_rate_limit_error(message: str) -> bool:
    m = message.lower()
    return "429" in m or "rate limit" in m or "quota" in m or "resource_exhausted" in m


def _is_not_found_error(message: str) -> bool:
    m = message.lower()
    return "not found" in m or "404" in m or "is not supported" in m


def _call_gemini(prompt: str) -> Dict[str, Any]:
    global _GEMINI_MODEL_CACHE

    api_key = _get_gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "provider": "GEMINI",
            "error": "GEMINI_API_KEY is not set",
            "model": None,
            "rate_limited": False,
        }

    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "provider": "GEMINI",
            "error": f"Gemini SDK import failed: {exc}",
            "model": None,
            "rate_limited": False,
        }

    genai.configure(api_key=api_key)

    candidates: List[str] = []
    if _GEMINI_MODEL_CACHE:
        candidates.append(_GEMINI_MODEL_CACHE)
    candidates.extend(m for m in GEMINI_MODEL_CANDIDATES if m not in candidates)

    errors: List[str] = []
    saw_rate_limit = False

    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                raise RuntimeError("Empty response")

            _GEMINI_MODEL_CACHE = model_name
            return {
                "ok": True,
                "provider": "GEMINI",
                "text": text,
                "model": model_name,
                "error": "",
                "rate_limited": False,
            }
        except Exception as exc:  # pragma: no cover
            msg = str(exc)
            errors.append(f"{model_name}: {msg}")
            if _is_rate_limit_error(msg):
                saw_rate_limit = True
                break
            if _is_not_found_error(msg):
                continue

    return {
        "ok": False,
        "provider": "GEMINI",
        "error": " | ".join(errors) if errors else "Gemini request failed",
        "model": _GEMINI_MODEL_CACHE,
        "rate_limited": saw_rate_limit,
    }


def _call_groq(prompt: str) -> Dict[str, Any]:
    key = _get_groq_api_key()
    if not key:
        return {
            "ok": False,
            "provider": "GROQ",
            "error": "GROQ_API_KEY (or Groq_API_KeY) is not set",
            "model": "llama-3.1-8b-instant",
        }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    req = urllib_request.Request(
        url="https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "llm-guardian/1.0",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            choices = parsed.get("choices", [])
            if not choices:
                return {
                    "ok": False,
                    "provider": "GROQ",
                    "error": "Empty response",
                    "model": payload["model"],
                }

            text = choices[0].get("message", {}).get("content", "")
            if not text:
                return {
                    "ok": False,
                    "provider": "GROQ",
                    "error": "Empty response",
                    "model": payload["model"],
                }

            return {
                "ok": True,
                "provider": "GROQ",
                "text": text,
                "model": payload["model"],
                "error": "",
            }
    except urllib_error.HTTPError as exc:  # pragma: no cover
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return {
            "ok": False,
            "provider": "GROQ",
            "error": detail,
            "model": payload["model"],
        }
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "provider": "GROQ",
            "error": str(exc),
            "model": payload["model"],
        }


def call_llm(prompt: str) -> Dict[str, Any]:
    gemini = _run_with_timeout(
        _call_gemini,
        prompt,
        timeout_sec=12,
        default={
            "ok": False,
            "provider": "GEMINI",
            "error": "Gemini request timed out",
            "model": _GEMINI_MODEL_CACHE,
            "rate_limited": False,
        },
    )

    if gemini.get("ok"):
        return {
            "llm_provider": "GEMINI",
            "llm_active_model": gemini.get("model"),
            "llm_response": gemini.get("text"),
            "gemini_response": gemini.get("text"),
            "groq_response": "[Groq Not Used: Gemini succeeded]",
        }

    groq = _run_with_timeout(
        _call_groq,
        prompt,
        timeout_sec=10,
        default={
            "ok": False,
            "provider": "GROQ",
            "error": "Groq request timed out",
            "model": "llama-3.1-8b-instant",
        },
    )

    gemini_error = f"[Gemini API Error: {gemini.get('error', 'Unknown error')}]"

    if groq.get("ok"):
        return {
            "llm_provider": "GROQ",
            "llm_active_model": groq.get("model"),
            "llm_response": groq.get("text"),
            "gemini_response": gemini_error,
            "groq_response": groq.get("text"),
        }

    groq_error = f"[Groq API Error: {groq.get('error', 'Unknown error')}]"
    combined = f"{gemini_error} | {groq_error}"
    return {
        "llm_provider": "NONE",
        "llm_active_model": None,
        "llm_response": combined,
        "gemini_response": gemini_error,
        "groq_response": groq_error,
    }


def _build_history_item(result: AnalysisResult) -> HistoryItem:
    preview = (result.original_prompt or "")[:80]
    return HistoryItem(
        event_id=result.event_id,
        timestamp=result.timestamp,
        decision=result.decision,
        risk_level=result.risk_level,
        risk_score=result.risk_score,
        prompt_preview=preview,
    )


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(
        INDEX_FILE,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/analyze", response_model=AnalysisResult)
def analyze_prompt(request: PromptRequest) -> AnalysisResult:
    recent_context: List[str] = list(CONTEXT_HISTORY)
    result = analyzer.analyze(request.prompt, history=recent_context)

    if result.decision in {"FORWARD", "SANITIZE"} and result.forwarded_prompt:
        llm_result = call_llm(result.forwarded_prompt)
        result.llm_provider = llm_result["llm_provider"]
        result.llm_active_model = llm_result["llm_active_model"]
        result.llm_response = llm_result["llm_response"]
        result.gemini_response = llm_result["gemini_response"]
        result.groq_response = llm_result["groq_response"]
    else:
        result.llm_provider = "NONE"
        result.llm_active_model = None
        result.llm_response = None
        result.gemini_response = None
        result.groq_response = None

    CONTEXT_HISTORY.append(request.prompt)
    ANALYSIS_EVENTS.append(result)

    LOGGER.info(
        "analysis event_id=%s decision=%s risk=%s score=%s provider=%s model=%s semantic=%.4f embedding=%.4f transformer=%.4f",
        result.event_id,
        result.decision,
        result.risk_level,
        result.risk_score,
        result.llm_provider,
        result.llm_active_model,
        result.semantic_similarity_score,
        result.embedding_similarity_score,
        result.transformer_injection_probability,
    )

    return result


@app.get("/history", response_model=List[HistoryItem])
def get_history() -> List[HistoryItem]:
    recent = list(ANALYSIS_EVENTS)[-50:]
    recent.reverse()
    return [_build_history_item(item) for item in recent]


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    events = list(ANALYSIS_EVENTS)
    total = len(events)

    blocked = sum(1 for e in events if e.decision == "BLOCK")
    sanitized = sum(1 for e in events if e.decision == "SANITIZE")
    forwarded = sum(1 for e in events if e.decision == "FORWARD")

    avg_risk = round(sum(e.risk_score for e in events) / total, 2) if total else 0.0

    high = sum(1 for e in events if e.risk_level == "HIGH")
    medium = sum(1 for e in events if e.risk_level == "MEDIUM")
    low = sum(1 for e in events if e.risk_level == "LOW")

    return StatsResponse(
        total_analyzed=total,
        total_blocked=blocked,
        total_sanitized=sanitized,
        total_forwarded=forwarded,
        average_risk_score=avg_risk,
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
    )


def open_browser() -> None:
    import webbrowser

    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
