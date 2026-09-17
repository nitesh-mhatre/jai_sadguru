"""
trading/async_llm.py
--------------------

Async multi-model LLM pool — runs multiple NVIDIA NIM models in parallel
so the trading loop never waits for a single slow model.

Architecture:
  • LLMPool owns N NvidiaAsyncClient instances (one per model)
  • .query_all(prompt) fires all models concurrently via httpx + asyncio
  • Returns as results arrive (first-to-respond can be used immediately)
  • .query_with_fallback(prompt) uses fastest model first, falls back if needed
  • .vote_all(prompt) — all models vote, majority wins (with confidence scoring)

Decoupled from trade execution: the main loop submits a prompt and gets back
a Future/result without blocking on any single network call.

Timeouts:
  • Each model has its own per-request timeout from config.NVIDIA_MODELS
  • The pool overall is capped at FAST_MODEL_TIMEOUT (120s) — if no model
    responds in time, the rule-based fallback is used instantly.

Usage:
    pool = LLMPool()
    # Fire all models in parallel, get results as they arrive
    results = await pool.query_all(prompt)
    for model_name, response in results:
        print(f"{model_name}: {response[:50]}...")

    # Fastest-response-first with fallback chain
    text = await pool.query_with_fallback(prompt)

    # Multi-model voting (majority direction wins)
    vote = await pool.vote_all(prompt)
    print(vote.direction, vote.confidence, vote.agreement)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config import (
    NVIDIA_BASE_URL,
    NVIDIA_MODELS,
    NVIDIA_KEYS,
    FAST_MODEL_TIMEOUT,
    VOTING_MODELS,
)

log = logging.getLogger(__name__)

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    """Single model response — arrives async as the call completes."""
    model_name:    str
    model_id:      str
    response_text: str = ""
    error:         str = ""
    elapsed:       float = 0.0
    success:       bool = False
    timestamp:     float = 0.0

    @property
    def direction(self) -> str:
        """Extract direction vote from JSON response if present."""
        if not self.response_text:
            return "UNKNOWN"
        try:
            data = json.loads(self.response_text)
            return str(data.get("direction", "")).upper()
        except (json.JSONDecodeError, AttributeError):
            return "UNKNOWN"

    @property
    def actions(self) -> list[dict]:
        """Extract actions list from JSON response if present."""
        if not self.response_text:
            return []
        try:
            data = json.loads(self.response_text)
            return data.get("actions", [])
        except (json.JSONDecodeError, AttributeError):
            return []

    @property
    def json_data(self) -> dict:
        """Parse full JSON response."""
        if not self.response_text:
            return {}
        try:
            return json.loads(self.response_text)
        except (json.JSONDecodeError, AttributeError):
            return {}


@dataclass
class VoteResult:
    """Aggregated vote from multiple models."""
    direction:      str = "UNKNOWN"     # BULLISH | BEARISH | SIDEWAYS | UNKNOWN
    confidence:     float = 0.0         # 0–1 agreement among models
    model_votes:    dict[str, str] = field(default_factory=dict)  # name→direction
    agreement_count: int = 0            # how many models agreed with majority
    total_models:   int = 0
    source:         str = "MULTI_MODEL_VOTE"

    @property
    def is_clear(self) -> bool:
        """True when there is strong agreement (>=70% of models agree)."""
        if self.total_models == 0:
            return False
        return (self.agreement_count / self.total_models) >= 0.7

    @property
    def majority_direction(self) -> str:
        return self.direction


# ── Async HTTP client per model ───────────────────────────────────────────────

class NvidiaAsyncClient:
    """
    Async HTTP client for a single NVIDIA NIM model.

    Uses httpx.AsyncClient for true async I/O — multiple clients can run
    concurrent requests without blocking each other.
    """

    def __init__(self, short_name: str, model_entry: dict):
        self.short_name = short_name
        self.model_id   = model_entry["model_id"]
        self.api_key    = model_entry.get("api_key", "") or ""
        self.max_tokens = model_entry.get("max_tokens", 2048)
        self.temperature = model_entry.get("temperature", 0.3)
        self.top_p      = model_entry.get("top_p", 1.0)
        self.stream     = model_entry.get("stream", False)
        self.timeout    = float(model_entry.get("timeout", FAST_MODEL_TIMEOUT))
        self.reasoning_effort = model_entry.get("reasoning_effort")

        # Per-model HTTP client — reuse connection pool
        self._client = httpx.AsyncClient(
            base_url=NVIDIA_BASE_URL,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, messages: list[dict], prompt: str = "",
                   tools: list[dict] | None = None) -> LLMResult:
        """
        Fire a single chat completion request. Returns LLMResult.

        Non-streaming only — streaming across multiple parallel models adds
        complexity with little benefit for the trading use case.
        """
        start = time.monotonic()
        result = LLMResult(
            model_name=self.short_name,
            model_id=self.model_id,
            timestamp=start,
        )

        payload: dict = {
            "model":      self.model_id,
            "messages":   messages,
            "temperature": self.temperature,
            "top_p":      self.top_p,
            "max_tokens": self.max_tokens,
            "stream":     False,
        }

        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        # If a system prompt suffix is needed, inject it
        if not messages or messages[0].get("role") != "system":
            payload["messages"] = [{"role": "system", "content": ""}] + messages

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

        try:
            resp = await self._client.post(
                "/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            result.elapsed = time.monotonic() - start

            if resp.status_code >= 500:
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                log.warning("[%s] HTTP %d — %s", self.short_name, resp.status_code,
                            resp.text[:150])
                return result

            if resp.status_code == 401:
                result.error = "HTTP 401: API key invalid"
                return result

            if resp.status_code == 429:
                result.error = "HTTP 429: rate limited"
                return result

            resp.raise_for_status()
            data = resp.json()

            choice = data.get("choices", [{}])[0]
            msg    = choice.get("message", {})
            content = msg.get("content", "") or ""

            # Extract tool calls if any
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and not content:
                # Model called tools — extract function results
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        content = json.dumps(json.loads(args))
                    except (json.JSONDecodeError, TypeError):
                        content = args

            result.response_text = content.strip()
            result.success = bool(content.strip()) or tool_calls

        except asyncio.TimeoutError:
            result.elapsed = time.monotonic() - start
            result.error = f"timeout ({self.timeout}s)"
            log.warning("[%s] timeout after %.1fs", self.short_name, result.elapsed)

        except httpx.TimeoutException:
            result.elapsed = time.monotonic() - start
            result.error = f"HTTP timeout ({self.timeout}s)"
            log.warning("[%s] HTTP timeout after %.1fs", self.short_name, result.elapsed)

        except httpx.RequestError as exc:
            result.elapsed = time.monotonic() - start
            result.error = f"request error: {exc}"
            log.warning("[%s] request error: %s", self.short_name, exc)

        except Exception as exc:
            result.elapsed = time.monotonic() - start
            result.error = str(exc)[:200]
            log.warning("[%s] unexpected error: %s", self.short_name, exc)

        return result


# ── Multi-model pool ───────────────────────────────────────────────────────────

class LLMPool:
    """
    Manages a pool of async LLM clients — one per model.

    Fire multiple models in parallel and aggregate results.

    Usage:
        pool = LLMPool(model_names=["llama-vision", "mistral", "nemo-light"])
        # Fire all in parallel
        results = await pool.query_all(prompt, messages)
        # Or get fastest response with fallback
        text = await pool.query_with_fallback(prompt, messages)
        # Or get a vote
        vote = await pool.vote_all(prompt, messages)
    """

    def __init__(self, model_names: list[str] | None = None):
        """
        Args:
            model_names: which models to include. Defaults to VOTING_MODELS
                         from config (llama-vision, mistral, nemo-light).
        """
        names = model_names or VOTING_MODELS
        self.clients: dict[str, NvidiaAsyncClient] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent = 3   # don't fire more than 3 at once per pool

        for name in names:
            entry = NVIDIA_MODELS.get(name)
            if entry:
                self.clients[name] = NvidiaAsyncClient(name, entry)
            else:
                log.warning("Model %s not found in NVIDIA_MODELS — skipped", name)

        log.info("LLMPool ready: %d models → %s", len(self.clients),
                 ", ".join(self.clients.keys()))

    async def close(self) -> None:
        """Close all HTTP clients."""
        await asyncio.gather(
            *[client.close() for client in self.clients.values()],
            return_exceptions=True,
        )

    async def query_all(self, prompt: str,
                        system_suffix: str = "",
                        timeout_override: float | None = None) -> list[LLMResult]:
        """
        Fire all models in parallel. Returns results in completion order.

        Args:
            prompt: user message text
            system_suffix: appended to system prompt
            timeout_override: cap total wait (defaults to FAST_MODEL_TIMEOUT)

        Returns:
            List of LLMResult sorted by elapsed time (fastest first).
        """
        timeout = timeout_override or FAST_MODEL_TIMEOUT
        messages = self._build_messages(prompt, system_suffix)

        # Fire all in parallel with a semaphore to limit concurrency
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _fire(client: NvidiaAsyncClient) -> LLMResult:
            async with self._semaphore:
                return await client.chat(messages)

        tasks = [asyncio.create_task(_fire(c)) for c in self.clients.values()]
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        results = [t.result() for t in done]
        # Sort by elapsed time — fastest first
        results.sort(key=lambda r: r.elapsed)
        return results

    async def query_with_fallback(self, prompt: str,
                                  system_suffix: str = "",
                                  timeout_override: float | None = None) -> LLMResult:
        """
        Query models in priority order (fastest first), return first success.

        If the fastest model fails or times out, immediately try the next one
        without waiting — true fallback, not retry.

        Args:
            prompt: user message text
            system_suffix: appended to system prompt
            timeout_override: per-model timeout cap

        Returns:
            First successful LLMResult, or last failure if all fail.
        """
        timeout = timeout_override or FAST_MODEL_TIMEOUT
        messages = self._build_messages(prompt, system_suffix)

        # Order by model speed: fastest first
        ordered = sorted(
            self.clients.values(),
            key=lambda c: c.timeout,
        )

        last_result = LLMResult(model_name="NONE", model_id="", error="no models available")

        for client in ordered:
            # Per-model timeout — don't wait longer than the model's own budget
            model_timeout = min(client.timeout, timeout)
            try:
                result = await asyncio.wait_for(
                    client.chat(messages),
                    timeout=model_timeout,
                )
                if result.success:
                    log.info("[%s] responded in %.2fs — using result",
                             client.short_name, result.elapsed)
                    return result
                # Non-success but no hard error — try next model
                log.warning("[%s] empty response — falling back to next model",
                            client.short_name)
            except asyncio.TimeoutError:
                log.warning("[%s] timeout — falling back to next model",
                            client.short_name)
            except Exception as exc:
                log.warning("[%s] error — falling back: %s", client.short_name, exc)

            last_result = LLMResult(
                model_name=client.short_name,
                model_id=client.model_id,
                error=f"failed, tried next: {client.short_name}",
            )

        return last_result

    async def vote_all(self, prompt: str,
                       system_suffix: str = "",
                       timeout_override: float | None = None) -> VoteResult:
        """
        Fire all models in parallel and aggregate their direction votes.

        Each model responds with JSON {"direction": "BULLISH|BEARISH|SIDEWAYS"}.
        Majority wins; confidence = agreement ratio.

        Args:
            prompt: user message text
            system_suffix: appended to system prompt
            timeout_override: total pool timeout

        Returns:
            VoteResult with aggregated direction + confidence.
        """
        timeout = timeout_override or FAST_MODEL_TIMEOUT
        messages = self._build_messages(prompt, system_suffix)

        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _fire(client: NvidiaAsyncClient) -> LLMResult:
            async with self._semaphore:
                return await client.chat(messages)

        tasks = {asyncio.create_task(_fire(c)): c.short_name
                 for c in self.clients.values()}

        done, pending = await asyncio.wait(
            tasks.keys(),
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        for task in pending:
            task.cancel()

        # Collect results
        model_votes: dict[str, str] = {}
        successful = 0

        for task in done:
            result = task.result()
            name = tasks[task]
            if result.success and result.direction in ("BULLISH", "BEARISH", "SIDEWAYS"):
                model_votes[name] = result.direction
                successful += 1
            else:
                model_votes[name] = "FAILED"

        # Count direction frequencies
        dir_counts: dict[str, int] = {}
        for d in model_votes.values():
            if d in ("BULLISH", "BEARISH", "SIDEWAYS"):
                dir_counts[d] = dir_counts.get(d, 0) + 1

        total = len(dir_counts)
        if total == 0:
            return VoteResult(
                direction="UNKNOWN",
                confidence=0.0,
                model_votes=model_votes,
                agreement_count=0,
                total_models=len(model_votes),
                source="MULTI_MODEL_VOTE (no valid responses)",
            )

        # Majority direction
        majority_dir = max(dir_counts, key=dir_counts.get)
        agreement = dir_counts[majority_dir]

        return VoteResult(
            direction=majority_dir,
            confidence=round(agreement / successful, 2) if successful else 0.0,
            model_votes=model_votes,
            agreement_count=agreement,
            total_models=successful,
            source="MULTI_MODEL_VOTE",
        )

    def _build_messages(self, prompt: str, system_suffix: str = "") -> list[dict]:
        """Build messages list from prompt + optional suffix."""
        suffix = system_suffix or ""
        return [{"role": "user", "content": prompt + ("\n\n" + suffix if suffix else "")}]


# ── Convenience: sync wrapper for use in non-async contexts ───────────────────

def run_async(coro) -> Any:
    """Run an async coroutine in a new event loop (for sync callers)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    else:
        # Already in an event loop — can't block, return coroutine
        return coro


def sync_query_with_fallback(pool: LLMPool, prompt: str,
                             system_suffix: str = "") -> LLMResult:
    """Sync wrapper — blocks until a model responds or all fail."""
    return run_async(pool.query_with_fallback(prompt, system_suffix))


def sync_vote_all(pool: LLMPool, prompt: str,
                  system_suffix: str = "") -> VoteResult:
    """Sync wrapper — blocks until all models vote or timeout."""
    return run_async(pool.vote_all(prompt, system_suffix))


# ── Singleton pool (lazy-init) ────────────────────────────────────────────────

_pool: Optional[LLMPool] = None

def get_pool(model_names: list[str] | None = None) -> LLMPool:
    """Get or create the global LLMPool singleton."""
    global _pool
    if _pool is None:
        _pool = LLMPool(model_names=model_names)
    return _pool
