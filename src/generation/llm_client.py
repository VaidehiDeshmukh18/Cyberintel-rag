"""Swappable LLM client for CyberIntel-RAG.

Reads `LLM_PROVIDER` from `.env` and routes `generate()` to the right
backend. Current providers:

    * ollama  — local server (default for this project)
    * groq    — Groq cloud API (free tier, Llama 3.1 70B)
    * openai  — OpenAI API (gpt-4o-mini etc.)

The returned `LLMResponse` is backend-agnostic: callers never see the
provider's native response shape. Swapping providers = one env change.

Design notes:
  * One retry on transient errors (connection refused, timeout). We do
    not retry on 4xx / content errors because they indicate a real bug.
  * Latency is measured server-side and attached to the response for
    monitoring in Phase 13.
  * `temperature=0.1` by default — grounded answers should be
    near-deterministic. Higher temps invite hallucination.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv
from loguru import logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

load_dotenv()  # pulls .env into os.environ on import

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
API_KEY = os.getenv("LLM_API_KEY", "").strip()
MODEL = os.getenv("LLM_MODEL", "llama3.2:latest").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Backend-agnostic response from any provider."""

    text: str                 # the generated answer
    model: str                # which model produced it
    provider: str             # which backend
    latency_s: float          # wall-clock time
    input_tokens: Optional[int] = None   # if the backend reports it
    output_tokens: Optional[int] = None  # if the backend reports it

    def __repr__(self) -> str:  # readable debug printing
        preview = self.text[:80].replace("\n", " ")
        return (
            f"LLMResponse(provider={self.provider!r} model={self.model!r} "
            f"latency={self.latency_s:.2f}s text={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_ollama(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> LLMResponse:
    """Call the local Ollama /api/chat endpoint."""
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
    latency = time.perf_counter() - t0

    text = body.get("message", {}).get("content", "").strip()
    return LLMResponse(
        text=text,
        model=model,
        provider="ollama",
        latency_s=latency,
        input_tokens=body.get("prompt_eval_count"),
        output_tokens=body.get("eval_count"),
    )


def _call_groq(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> LLMResponse:
    """Call Groq's OpenAI-compatible chat completions API."""
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY not set for provider=groq")

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
    latency = time.perf_counter() - t0

    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    return LLMResponse(
        text=text,
        model=model,
        provider="groq",
        latency_s=latency,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def _call_openai(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> LLMResponse:
    """Call OpenAI's chat completions API."""
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY not set for provider=openai")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
    latency = time.perf_counter() - t0

    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    return LLMResponse(
        text=text,
        model=model,
        provider="openai",
        latency_s=latency,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


_PROVIDERS = {
    "ollama": _call_ollama,
    "groq": _call_groq,
    "openai": _call_openai,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    provider: str | None = None,
) -> LLMResponse:
    """Call the configured LLM and return a typed response.

    Params `model` and `provider` override the .env defaults (useful for
    evaluation runs comparing providers).
    """
    provider = (provider or PROVIDER).lower()
    model = model or MODEL
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Supported: {sorted(_PROVIDERS)}"
        )

    fn = _PROVIDERS[provider]

    # One retry on transient network errors — common with local Ollama
    # when the model is being paged into RAM for the first time.
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            return fn(system, user, model, temperature, max_tokens, timeout_s)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exc = e
            logger.warning(
                f"LLM call failed (attempt {attempt}/2) via {provider}: {e}"
            )
            time.sleep(1.0)
        except httpx.HTTPStatusError as e:
            # 4xx/5xx — don't retry. Something's actually wrong.
            logger.error(
                f"LLM call rejected by {provider} ({e.response.status_code}): "
                f"{e.response.text[:200]}"
            )
            raise

    raise RuntimeError(f"LLM call failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    system = "You are a cybersecurity expert. Keep answers to one sentence."
    user = "What is CVE-2021-44228 in one sentence?"

    logger.info(f"Testing provider={PROVIDER!r} model={MODEL!r}")
    resp = generate(system=system, user=user)

    print()
    print(f"Provider:       {resp.provider}")
    print(f"Model:          {resp.model}")
    print(f"Latency:        {resp.latency_s:.2f} s")
    print(f"Input tokens:   {resp.input_tokens}")
    print(f"Output tokens:  {resp.output_tokens}")
    print()
    print("ANSWER:")
    print(resp.text)


if __name__ == "__main__":
    main()