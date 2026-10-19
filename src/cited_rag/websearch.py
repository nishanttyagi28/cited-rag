"""Pluggable web search used as a fallback when corpus retrieval confidence is low.

Off by default (Config.web_fallback=False). Providers:
  fake   : returns canned results, used in tests and demos
  tavily : https://tavily.com, needs TAVILY_API_KEY
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


class WebSearchProvider(Protocol):
    name: str

    def search(self, query: str, k: int = 3) -> list[WebResult]: ...


class FakeWebSearch:
    name = "fake"

    def __init__(self, results: list[WebResult] | None = None) -> None:
        self.results = results or [
            WebResult("Example result", "https://example.org/a", "Placeholder web snippet.")
        ]
        self.queries: list[str] = []

    def search(self, query: str, k: int = 3) -> list[WebResult]:
        self.queries.append(query)
        return self.results[:k]


class TavilySearch:
    name = "tavily"
    url = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, timeout: float = 20.0) -> None:
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self._key = key
        self.timeout = timeout

    def search(self, query: str, k: int = 3) -> list[WebResult]:
        body = json.dumps({"query": query, "max_results": k}).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return [
            WebResult(r.get("title", ""), r.get("url", ""), r.get("content", ""))
            for r in data.get("results", [])[:k]
        ]


PROVIDERS = {"fake": FakeWebSearch, "tavily": TavilySearch}


def get_provider(name: str) -> WebSearchProvider:
    try:
        return PROVIDERS[name]()
    except KeyError as e:
        raise ValueError(f"unknown web provider {name!r}; known: {sorted(PROVIDERS)}") from e
