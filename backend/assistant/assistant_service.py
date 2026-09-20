"""Assistant service: intent detection, tool dispatch, LLM call.

When an LLM provider/credentials are configured (env vars), the question plus
the real backend context is sent to the provider and the answer is returned.
When no provider is configured, ``prompts.build_fallback`` produces an honest
template answer from the same tool data. Either way the frontend and the API
get: answer, sources, llm status, demo flag, warnings.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
import uuid
from typing import Any

from assistant import prompts
from assistant.tools import TOOLS, datasets_status, extracted_iceberg_ids
from config import settings

logger = logging.getLogger("dss.assistant")

TIMEOUT_S = 45
MAX_TOKENS = 800

DEFAULT_ROUTE = {
    "start_lat": -65.0,
    "start_lon": 140.0,
    "dest_lat": -77.8469,
    "dest_lon": 166.6687,
    "vessel_id": settings.DEFAULT_VESSEL_ID,
    "preference": "recommended",
}


class AssistantError(Exception):
    """Raised when the assistant cannot produce any answer at all."""


def resolve_provider() -> dict:
    """Return (provider, model) according to config/env; never the API key."""
    cfg = settings.ASSISTANT_PROVIDER.strip().lower()
    model_override = (settings.ASSISTANT_MODEL or "").strip() or None

    if cfg == "none":
        return {"provider": "none", "model": None, "configured": False}
    if cfg in ("openai", "anthropic", "ollama"):
        provider = cfg
        model = model_override
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                return {"provider": "none", "model": None, "configured": False, "reason": f"{cfg} requested but OPENAI_API_KEY missing"}
            model = model or settings.OPENAI_MODEL
        elif provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                return {"provider": "none", "model": None, "configured": False, "reason": f"{cfg} requested but ANTHROPIC_API_KEY missing"}
            model = model or settings.ANTHROPIC_MODEL
        else:  # ollama
            model = model or settings.OLLAMA_MODEL
        return {"provider": provider, "model": model, "configured": True}

    # auto: only providers with actual credentials are auto-picked.
    # Ollama has no key, so it needs ASSISTANT_PROVIDER=ollama explicitly.
    if settings.OPENAI_API_KEY:
        return {"provider": "openai", "model": model_override or settings.OPENAI_MODEL, "configured": True}
    if settings.ANTHROPIC_API_KEY:
        return {"provider": "anthropic", "model": model_override or settings.ANTHROPIC_MODEL, "configured": True}
    return {"provider": "none", "model": None, "configured": False}


# --------------------------------------------------------------------------
# LLM clients (stdlib only, credentials in-process)
# --------------------------------------------------------------------------
def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        **({} if headers is None else headers),
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310 - configured by the operator
        return json.loads(resp.read().decode("utf-8"))


def _chat_openai(model: str, messages: list[dict]) -> str:
    base = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    data = _post_json(
        f"{base}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
        },
        {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
    )
    return data["choices"][0]["message"]["content"]


def _chat_anthropic(model: str, system: str, history: list[dict]) -> str:
    msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "system": system,
            "messages": msgs,
            "max_tokens": MAX_TOKENS,
        },
        {
            "x-api-key": settings.ANTHROPIC_API_KEY or "",
            "anthropic-version": "2023-06-01",
        },
    )
    return "".join(block.get("text", "") for block in data.get("content", []))


def _chat_ollama(model: str, messages: list[dict]) -> str:
    base = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    data = _post_json(
        f"{base}/api/chat",
        {"model": model, "messages": messages, "stream": False},
    )
    return data.get("message", {}).get("content", "")


def run_llm(provider: str, model: str, messages: list[dict]) -> str:
    """Call the configured provider. Raises AssistantError on failure."""
    try:
        if provider == "openai":
            return _chat_openai(model, messages)
        if provider == "anthropic":
            system = messages[0]["content"]
            return _chat_anthropic(model, system, messages[1:])
        if provider == "ollama":
            return _chat_ollama(model, messages)
    except (urllib.error.URLError, KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
        raise AssistantError(f"LLM provider {provider} ({model}) error: {exc}") from exc
    raise AssistantError(f"Unknown LLM provider: {provider}")


# --------------------------------------------------------------------------
# Intent detection -> which tools to run
# --------------------------------------------------------------------------
def _pick_tools(question: str, horizon: int) -> list[tuple[str, dict]]:
    q = question.lower()
    tools: list[tuple[str, dict]] = []
    ids = extracted_iceberg_ids(question)

    if ids:
        a, b = ids[0], (ids[1] if len(ids) > 1 else None)
        if "distance" in q:
            tools.append(("iceberg_distance", {"iceberg_a": a, "iceberg_b": b or a}))
        else:
            tools.append(("iceberg_trajectory", {"iceberg_id": a, "horizon_hours": horizon}))

    if any(k in q for k in ("sea-ice", "sea ice", "concentration", "sea_ice")):
        if any(k in q for k in ("actual", "current", "observed", "difference")):
            tools.append(("sea_ice_current", {}))
            if "difference" in q or "actual" in q:
                tools.append(("sea_ice_forecast", {"horizon_hours": horizon}))
        else:
            tools.append(("sea_ice_forecast", {"horizon_hours": horizon}))

    if "forecast" in q or "prediction" in q or "accuracy" in q:
        tools.append(("model_metrics", {"pipeline": "sea_ice" if "sea" in q else None}))

    if "route" in q or "fuel" in q or "risk" in q:
        tools.append(("route_details", dict(DEFAULT_ROUTE)))
    if "fuel" in q:
        tools.append(("fuel_estimate", dict(DEFAULT_ROUTE)))

    if "closest" in q or "nearest" in q:
        tools.append(("closest_iceberg", {}))

    if "how many" in q and "iceberg" in q:
        tools.append(("iceberg_list", {}))

    if not tools:
        tools.append(("datasets_status", {}))
        if "iceberg" in q and "drift" not in q:
            tools.append(("iceberg_list", {}))
        if "drift" in q:
            tools.append(("iceberg_trajectory", {"iceberg_id": ids[0] if ids else "DEMO-B000", "horizon_hours": horizon}))
    return tools


class AssistantService:
    """Stateless assistant: answer(question, history) -> response dict."""

    def llm_status(self) -> dict:
        return {k: v for k, v in resolve_provider().items() if k != "reason"}

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        horizon_hours: int = 24,
    ) -> dict:
        horizon = max(2, min(168, int(horizon_hours)))
        provider_info = resolve_provider()
        provider = provider_info["provider"]

        tool_results = [
            TOOLS[name](**kwargs)
            for name, kwargs in _pick_tools(question, horizon)
        ]
        # always know whether the data is real or demo
        if not any(r.get("name") == "datasets_status" for r in tool_results):
            tool_results.append(datasets_status())

        tool_results = [r for r in tool_results if r is not None]
        context_report = prompts.build_context_report(tool_results)

        sources = [
            {
                "name": r["name"],
                "demo": bool(r.get("demo")),
                "note": r.get("note"),
                "status": r.get("status", "ok"),
            }
            for r in tool_results
        ]

        demo_any = any(r.get("demo") for r in tool_results if r.get("status") == "ok")
        real_any = any((not r.get("demo")) for r in tool_results if r.get("status") == "ok")
        warnings: list[str] = []

        if provider == "none":
            answer = prompts.build_fallback(question, tool_results)
            llm = {"provider": "none", "model": None, "configured": False}
            llm_note = provider_info.get("reason")
            if llm_note:
                warnings.append(llm_note)
        else:
            messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                *prompts.build_history_prefix(history),
                {"role": "user", "content": prompts.build_user_prompt(question, context_report)},
            ]
            try:
                text = run_llm(provider, provider_info["model"], messages).strip()
                answer = text or prompts.build_fallback(question, tool_results)
                llm = {"provider": provider, "model": provider_info["model"], "configured": True}
            except AssistantError as exc:
                logger.warning("LLM call failed, falling back to template: %s", exc)
                answer = prompts.build_fallback(question, tool_results)
                llm = {"provider": provider, "model": provider_info["model"], "configured": True}
                warnings.append(f"LLM call failed ({exc}); answered from templates instead.")

        if demo_any and not real_any:
            warnings.append(
                "All data behind this answer is synthetic demo data. It carries no "
                "information about real-world conditions and must not be used for "
                "actual navigation decisions."
            )
        if real_any:
            warnings.append(
                "Predictions are model estimates; no route is guaranteed safe."
            )

        return {
            "answer": answer,
            "sources": sources,
            "llm": llm,
            "demo_mode": demo_any,
            "warnings": warnings,
        }


assistant_service = AssistantService()