import sys
import asyncio
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def api_base(url: str) -> str:
    """Accept the server root, its /v1, or a copied /docs URL."""
    url = url.strip().rstrip("/")
    for suffix in ("/docs", "/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"

class RequestFailed(Exception):
    pass


def parse_base_urls(value: str) -> List[str]:
    """One or more server URLs: a JSON list (``["http://a:1812", "http://a:1813"]``), a comma- or
    space-separated list, or a single URL. Each is normalised with ``api_base``."""
    value = (value or "").strip()
    if value.startswith("["):
        urls = json.loads(value)
    else:
        urls = [u for u in re.split(r"[,\s]+", value) if u]
    urls = list(dict.fromkeys(api_base(u) for u in urls))
    if not urls:
        raise ValueError("no server URL in %r" % value)
    return urls


class Endpoint:
    """One vLLM server: its own HTTP client and health state."""

    def __init__(self, url: str, http, capacity: int):
        self.url, self.http, self.capacity = url, http, capacity
        self.sent = self.failed = self.consecutive_failures = 0
        self.cooling_until = 0.0

    def cooling(self, now: float) -> bool:
        return now < self.cooling_until


class SlotPool:
    """``capacity`` request slots per server in one queue.

    A request takes whichever slot frees up first, so a faster or less loaded server serves more
    requests, and gives it back when done. Slots of a server that is cooling down after repeated
    failures are skipped while any healthy server exists."""

    def __init__(self, endpoints: List[Endpoint]):
        self.endpoints = endpoints
        self.queue: "asyncio.Queue[int]" = asyncio.Queue()
        for slot in range(max(e.capacity for e in endpoints)):
            for i, e in enumerate(endpoints):  # interleaved, so the first requests spread over servers
                if slot < e.capacity:
                    self.queue.put_nowait(i)

    async def acquire(self) -> Endpoint:
        loop = asyncio.get_running_loop()
        while True:
            i = await self.queue.get()
            ep, now = self.endpoints[i], loop.time()
            if not ep.cooling(now) or all(e.cooling(now) for e in self.endpoints):
                return ep
            self.queue.put_nowait(i)
            await asyncio.sleep(0.05)

    def release(self, ep: Endpoint) -> None:
        self.queue.put_nowait(self.endpoints.index(ep))


class VLLMClient:
    """OpenAI-compatible client over one or more vLLM servers serving the same model."""

    FAILURES_BEFORE_COOLDOWN = 3
    COOLDOWN_SECONDS = 30.0

    def __init__(self, base_url, model: str,  *, api_key: Optional[str] = None,
                 max_concurrent: int = 16, top_logprobs: int = 20, timeout: float = 300.0, retries: int = 4,
                 transport=None):
        """``base_url``: a URL, a list of URLs, or anything ``parse_base_urls`` accepts.
        ``max_concurrent`` is per server. ``transport`` is for tests (an ``httpx`` mock transport)."""
        import httpx  # lazy: keeps the module importable for offline tests

        urls = parse_base_urls(base_url) if isinstance(base_url, str) else [api_base(u) for u in base_url]
        self.model, self.top_logprobs, self.retries = model, top_logprobs, retries
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.endpoints = [Endpoint(u, httpx.AsyncClient(timeout=timeout, headers=headers, transport=transport,
                                                        limits=httpx.Limits(max_connections=max_concurrent)),
                                   max_concurrent) for u in urls]
        self.pool = SlotPool(self.endpoints)

    async def check_models(self) -> None:
        """Every server must be reachable and serve ``self.model``: the cache is keyed by model name,
        so a server with another model would mix two teachers' answers."""
        problems = []
        for ep in self.endpoints:
            try:
                resp = await ep.http.get(ep.url + "/models")
                ids = [m.get("id") for m in resp.json().get("data", [])]
                if self.model not in ids:
                    problems.append("%s serves %s, not %s" % (ep.url, ids, self.model))
            except Exception as e:
                problems.append("%s unreachable (%s: %s)" % (ep.url, type(e).__name__, e))
        if problems:
            raise RequestFailed("; ".join(problems))

    def endpoint_stats(self) -> List[Dict[str, Any]]:
        return [{"url": e.url, "sent": e.sent, "failed": e.failed} for e in self.endpoints]

    def _failed(self, ep: Endpoint) -> None:
        ep.failed += 1
        ep.consecutive_failures += 1
        if ep.consecutive_failures >= self.FAILURES_BEFORE_COOLDOWN and len(self.endpoints) > 1:
            ep.cooling_until = asyncio.get_running_loop().time() + self.COOLDOWN_SECONDS
            ep.consecutive_failures = 0
            print("  %s failing; sending its requests to the other server(s) for %.0fs"
                     % (ep.url, self.COOLDOWN_SECONDS), file=sys.stderr, flush=True)

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to whichever server has a free slot. Retryable failures retry with a new slot, which
        may be on another server; with a single server the waits back off exponentially."""
        import httpx

        delay = 2.0
        for attempt in range(self.retries + 1):
            ep = await self.pool.acquire()
            try:
                try:
                    resp = await ep.http.post(ep.url + "/chat/completions", json=payload)
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    # Transport failure (server down, refused, timeout):
                    # retryable - the next attempt acquires a new slot,
                    # which may land on another server.
                    self._failed(ep)
                    if attempt == self.retries:
                        raise RequestFailed("%s: %s: %s" % (ep.url, type(e).__name__, e)) from e
                else:
                    if resp.status_code == 200:
                        ep.sent += 1
                        ep.consecutive_failures = 0
                        ep.cooling_until = 0.0  # a live response ends any cooldown
                        return resp.json()
                    if resp.status_code not in RETRYABLE_STATUS:  # the request's fault, not the server's
                        raise RequestFailed("%s: HTTP %d: %s" % (ep.url, resp.status_code, resp.text[:300]))
                    self._failed(ep)
                    if attempt == self.retries:
                        raise RequestFailed("%s: HTTP %d: %s" % (ep.url, resp.status_code, resp.text[:300]))
            finally:
                self.pool.release(ep)
            wait = 0.5 if len(self.endpoints) > 1 else delay
            print("  request retry %d in %.1fs" % (attempt + 1, wait), file=sys.stderr, flush=True)
            await asyncio.sleep(wait)
            delay *= 2
        raise RequestFailed("unreachable")

    async def text_completion(self, messages: List[Dict[str, str]], *, max_tokens: int = 1024, thinking: bool = False, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        """A text answer (used for CoT rewriting)."""
        payload = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if seed is not None:
            payload["seed"] = seed

        def extract(body):
            return body["choices"][0]["message"]["content"] or ""

        return extract(await self._post(payload))

    async def aclose(self) -> None:
        for ep in self.endpoints:
            await ep.http.aclose()
