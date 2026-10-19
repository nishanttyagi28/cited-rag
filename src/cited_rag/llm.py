"""LLM clients. Groq (OpenAI-compatible HTTP API) for real runs, FakeLLM for tests.

The Groq free tier allows ~8k tokens per minute, so GroqLLM paces itself with a token
bucket and retries on 429 (rate limit) and 5xx. A 413 (request too large) is raised as
RequestTooLarge so the caller can shrink the context and try again.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class RequestTooLarge(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


class TokenPacer:
    """Sliding one-minute window over estimated token spend."""

    def __init__(self, tokens_per_minute: int, clock=time.monotonic, sleep=time.sleep) -> None:
        self.tpm = tokens_per_minute
        self._events: deque[tuple[float, int]] = deque()
        self._clock, self._sleep = clock, sleep

    def wait(self, tokens: int) -> float:
        tokens = min(tokens, self.tpm)
        waited = 0.0
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= 60:
                self._events.popleft()
            used = sum(n for _, n in self._events)
            if used + tokens <= self.tpm:
                self._events.append((now, tokens))
                return waited
            delay = 60 - (now - self._events[0][0]) + 0.05
            self._sleep(delay)
            waited += delay


class GroqLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        tokens_per_minute: int | None = None,
        max_retries: int = 6,
        timeout: float = 90.0,
    ) -> None:
        self.model = model or os.environ.get("CITED_RAG_MODEL", DEFAULT_MODEL)
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set")
        self._key = key
        tpm = tokens_per_minute or int(os.environ.get("CITED_RAG_TPM", "7000"))
        self.pacer = TokenPacer(tpm)
        self.max_retries = max_retries
        self.timeout = timeout
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
                "User-Agent": "cited-rag/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_completion_tokens": max_tokens,
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"
        budget = estimate_tokens(system + user) + max_tokens
        for attempt in range(self.max_retries + 1):
            self.pacer.wait(budget)
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as e:
                if e.code == 413:
                    raise RequestTooLarge(f"request of ~{budget} tokens rejected") from e
                if e.code == 429 or e.code >= 500:
                    if attempt == self.max_retries:
                        raise LLMError(f"giving up after {attempt + 1} tries: HTTP {e.code}") from e
                    retry_after = e.headers.get("retry-after") if e.headers else None
                    delay = float(retry_after) if retry_after else min(60.0, 2.0 ** (attempt + 1))
                    time.sleep(delay + random.uniform(0, 0.5))
                    continue
                raise LLMError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise LLMError(str(e)) from e
                time.sleep(2.0 ** (attempt + 1))
                continue
            usage = data.get("usage", {})
            self.usage["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                self.usage[k] += int(usage.get(k, 0))
            return (data["choices"][0]["message"].get("content") or "").strip()
        raise LLMError("unreachable")


class FakeLLM:
    """Deterministic stand-in. ``responder(system, user) -> str`` decides the reply."""

    def __init__(self, responder: Callable[[str, str], str] | str = "NO_ANSWER") -> None:
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        self.calls.append((system, user))
        if callable(self._responder):
            return self._responder(system, user)
        return self._responder
