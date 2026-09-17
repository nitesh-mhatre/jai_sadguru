"""
agent.py
--------
Agentic reasoning loop using NVIDIA NIM (OpenAI-compatible) with native
tool/function calling. Ollama support removed — NVIDIA NIM is the only backend.

The agent yields typed AgentEvent objects (UI-agnostic).

Event flow per user turn:
  User message
      │
      ▼
  NVIDIA NIM /v1/chat/completions (with tools JSON)
      │
      ├─ tool_calls in response → dispatch tool → inject result → loop
      └─ content text           → yield FinalAnswerEvent → return
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Generator

import requests
from openai import OpenAI as _OpenAI

from config import (
    SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    NVIDIA_BASE_URL,
    Config,
)
from tools import TOOL_SPECS, dispatch

log = logging.getLogger(__name__)

# 12 is plenty for the 2-round tool flow in SYSTEM_PROMPT. The old 25 let
# broken loops burn minutes before the user saw anything.
MAX_TOOL_ITERATIONS = 12

# Per-request cap. If a single NIM call takes >45s something is wrong —
# fail fast with a clear message instead of an endless spinner.
REQUEST_TIMEOUT = 45.0


# ── Event types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallEvent:
    name:        str
    params:      dict
    tool_call_id: str

@dataclass(frozen=True)
class ToolResultEvent:
    name:     str
    result:   str
    is_error: bool

@dataclass(frozen=True)
class FinalAnswerEvent:
    text: str

@dataclass(frozen=True)
class ErrorEvent:
    message: str

@dataclass(frozen=True)
class TokenEvent:
    """Incremental model output — lets the UI show live progress."""
    text: str

AgentEvent = ToolCallEvent | ToolResultEvent | FinalAnswerEvent | ErrorEvent | TokenEvent


# ── Session / history ─────────────────────────────────────────────────────────

@dataclass
class Session:
    messages: list[dict] = field(default_factory=list)

    def add(self, role: str, content) -> None:
        self.messages.append({"role": role, "content": content})

    def trim(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        if len(self.messages) > max_turns * 2:
            self.messages = self.messages[-(max_turns * 2):]

    def clear(self) -> None:
        self.messages.clear()


# ── JSON extraction helper ────────────────────────────────────────────────────

def extract_json(text: str):
    """
    Extract the first JSON object/array from model text.
    Handles ```json fences, ``` fences, and bare JSON.
    Returns None when no valid JSON is found.
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    for candidate in (raw, raw.strip("` \n")):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    # Last resort: strip trailing commas
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    except Exception:
        return None


# ── NVIDIA NIM client ─────────────────────────────────────────────────────────

class NvidiaClient:
    """
    Thin wrapper around the NVIDIA NIM API using the OpenAI-compatible SDK.

    Each model in NVIDIA_MODELS has its own api_key so all keys are used
    automatically depending on which model is selected.
    """

    def __init__(self, model: str, temperature: float, top_p: float,
                 max_tokens: int, api_key: str = "",
                 reasoning_effort: str | None = None,
                 timeout: float | None = None,
                 stream: bool = True):
        self.model       = model
        self.temperature = temperature
        self.top_p       = top_p
        self.max_tokens  = max_tokens
        self.reasoning_effort = reasoning_effort
        # Streaming is the default because non-streaming made the CLI look
        # frozen. Some models, however, break on stream=True + tools (mistral-
        # nemotron answers 500, others never emit a finish chunk), so the model
        # entry can turn it off. See config.NVIDIA_MODELS.
        self.stream      = stream
        self.on_token = None            # optional callback(delta_text) for live UI
        _key = api_key or "nvapi-SET-NVIDIA_API_KEY"
        # Per-model timeout: a slow reasoning model needs a long budget, while a
        # fast one should surface a failure quickly. A single global value meant
        # glm-5.3 (needs >75s) ALWAYS blew the 45s cap — the "glm unavailable"
        # error users saw.
        self.timeout = float(timeout or REQUEST_TIMEOUT)
        self._client = _OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=_key,
            timeout=self.timeout,
            max_retries=1,   # 1 retry only — 2 retries made failures take minutes
        )

    def check_connection(self) -> tuple[bool, str]:
        try:
            models = self._client.models.list()
            names  = [m.id for m in models.data][:5]
            return True, f"NVIDIA NIM connected. Models: {', '.join(names)}…"
        except Exception as exc:
            msg = str(exc)
            if "401" in msg:
                msg += "  → API key invalid/expired. Update NVIDIA_KEYS in config.py."
            return False, f"NVIDIA NIM connection failed: {msg}"

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Call NVIDIA NIM (streaming) and return a response envelope:
          {"message": {"role","content","tool_calls"}, "done": True}

        Streaming matters: non-streaming calls hold the whole response until
        generation finishes, which made the CLI look frozen for minutes.
        """
        kwargs: dict = dict(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            top_p       = self.top_p,
            max_tokens  = self.max_tokens,
            stream      = True,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Models flagged stream=False skip the streaming path entirely.
        if not self.stream:
            kwargs["stream"] = False
            return self._chat_once(kwargs)

        def _create(kw: dict):
            return self._client.chat.completions.create(**kw)

        try:
            stream = _create(kwargs)
        except Exception as exc:
            msg = str(exc)
            # Some reasoning models reject temperature/top_p — retry bare
            if any(k in msg.lower() for k in ("temperature", "top_p", "top-p", "unsupported")):
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                stream = _create(kwargs)
            elif "reasoning_effort" in msg:
                kwargs.pop("reasoning_effort", None)
                stream = _create(kwargs)
            else:
                raise

        # Stream opened — consume it, but never let a broken stream kill the
        # turn: some NIM models accept stream=True and then go silent.
        try:
            return self._consume_stream(stream)
        except Exception as exc:
            log.warning("stream consumption failed (%s) — retrying non-streaming",
                        str(exc)[:90])
            return self._chat_once({**kwargs, "stream": False})

    def _chat_once(self, kwargs: dict) -> dict:
        """
        Non-streaming call — used when streaming is disabled or fails.
        Returns the same response envelope as the streaming path.
        """
        kw = dict(kwargs)
        kw["stream"] = False
        try:
            resp = self._client.chat.completions.create(**kw)
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("temperature", "top_p", "top-p", "unsupported")):
                kw.pop("temperature", None)
                kw.pop("top_p", None)
                resp = self._client.chat.completions.create(**kw)
            elif "reasoning_effort" in msg:
                kw.pop("reasoning_effort", None)
                resp = self._client.chat.completions.create(**kw)
            else:
                raise

        msg        = resp.choices[0].message
        tool_calls: list[dict] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn   = getattr(tc, "function", None)
            args = getattr(fn, "arguments", "") if fn else ""
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id":       getattr(tc, "id", None) or f"call_{len(tool_calls)}",
                "type":     "function",
                "function": {
                    "name":      getattr(fn, "name", "") if fn else "",
                    "arguments": args if isinstance(args, dict) else {},
                },
            })
        return {
            "message": {
                "role":       "assistant",
                "content":    getattr(msg, "content", "") or "",
                "tool_calls": tool_calls,
            },
            "done": True,
        }

    def _consume_stream(self, stream) -> dict:
        """Read an SSE stream into the standard response envelope."""
        # ── Consume the stream ────────────────────────────────────────────
        content = ""
        tool_calls_raw: dict[int, dict] = {}
        finish_reason = None

        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content += delta.content
                if self.on_token:
                    try:
                        self.on_token(delta.content)
                    except Exception:
                        pass
            tcs = getattr(delta, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    idx = getattr(tc, "index", 0) or 0
                    acc = tool_calls_raw.setdefault(idx, {
                        "id":       getattr(tc, "id", None) or f"call_{idx}",
                        "type":     "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if getattr(tc, "id", None):
                        acc["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            acc["function"]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            acc["function"]["arguments"] += fn.arguments

        # ── Normalize tool calls (arguments str → dict) ───────────────────
        tool_calls: list[dict] = []
        for tc in tool_calls_raw.values():
            args = tc["function"].get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id":       tc.get("id") or f"call_{len(tool_calls)}",
                "type":     "function",
                "function": {
                    "name":      tc["function"]["name"],
                    "arguments": args if isinstance(args, dict) else {},
                },
            })

        return {
            "message": {
                "role":       "assistant",
                "content":    content,
                "tool_calls": tool_calls,
            },
            "done": True,
        }


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Stateless executor: state lives entirely in Session and Config.
    Call run() once per user turn.
    """

    def __init__(self, config: Config) -> None:
        from config import NVIDIA_MODELS, NVIDIA_DEFAULT_MODEL
        short = config.nvidia_model
        entry = NVIDIA_MODELS.get(short)
        if entry is None:
            # Try matching by full model_id
            entry = next(
                (v for v in NVIDIA_MODELS.values() if v["model_id"] == short),
                NVIDIA_MODELS[NVIDIA_DEFAULT_MODEL],
            )
        self._client = NvidiaClient(
            model       = entry["model_id"],
            temperature = entry["temperature"],
            top_p       = entry["top_p"],
            max_tokens  = entry["max_tokens"],
            api_key     = entry["api_key"],
            reasoning_effort = entry.get("reasoning_effort"),
            timeout     = entry.get("timeout"),
            stream      = entry.get("stream", True),
        )
        self._model_label = short
        self._config = config
        self._failover_used = False   # one automatic model failover per Agent

    def _switch_to_backup(self) -> bool:
        """
        Switch the active client to the failover model (NVIDIA_FAILOVER_MODEL).
        Returns True if the switch happened. Only once per Agent instance —
        if the backup also fails, the error surfaces to the user.
        """
        if self._failover_used:
            return False
        from config import NVIDIA_MODELS, NVIDIA_FAILOVER_MODEL
        name   = NVIDIA_FAILOVER_MODEL
        backup = NVIDIA_MODELS.get(name)
        if not backup or self._client.model == backup["model_id"]:
            return False
        log.warning("Failing over %s → %s", self._client.model, name)
        self._client = NvidiaClient(
            model       = backup["model_id"],
            temperature = backup["temperature"],
            top_p       = backup["top_p"],
            max_tokens  = backup["max_tokens"],
            api_key     = backup["api_key"],
            timeout     = backup.get("timeout"),
            stream      = backup.get("stream", True),
        )
        self._model_label = f"{name} (failover)"
        self._failover_used = True
        return True

    def check_health(self) -> tuple[bool, str]:
        return self._client.check_connection()

    def run(
        self,
        user_message: str,
        session: Session,
        system_suffix: str = "",
    ) -> Generator[AgentEvent, None, None]:
        """Execute one user turn, yielding events. Always calls tools fresh."""
        session.trim()

        # ── Inject fresh market time + regime context into every call ──────────
        time_context = ""
        try:
            from trading.market_time import get_market_status
            from trading.market_regime import REGIME_TRADING_RULES
            ms           = get_market_status()
            time_context = "\n\n" + ms.context_block() + "\n" + REGIME_TRADING_RULES
            t = ms.now_ist.time()
            from datetime import time as _time
            if _time(9, 0) <= t <= _time(9, 30):
                from trading.opening_momentum import OPENING_MOMENTUM_PROMPT
                time_context += OPENING_MOMENTUM_PROMPT
        except Exception as exc:
            log.warning("market_time inject failed: %s", exc)

        # ── Inject core-managed rules (learned in simulation) ─────────────────
        try:
            from trading.rules import load_rules_block
            time_context += "\n\n" + load_rules_block()
        except Exception as exc:
            log.warning("rules inject failed: %s", exc)

        system_content = SYSTEM_PROMPT + time_context + (
            "\n\n" + system_suffix if system_suffix else ""
        )

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(session.messages)
        messages.append({"role": "user", "content": user_message})

        active_tools = TOOL_SPECS
        iterations    = 0
        _empty_retries = 0              # recover dropped/empty model responses
        _tool_history: list[str] = []   # track "tool_name:key_param" for loop detection

        while iterations < MAX_TOOL_ITERATIONS:
            iterations += 1
            log.debug("Iteration %d  messages=%d", iterations, len(messages))

            try:
                response = self._client.chat(messages, active_tools)
            except Exception as exc:
                msg = str(exc)
                lowered = msg.lower()
                retriable = (
                    "timed out" in lowered or "timeout" in lowered
                    or "rate limit" in lowered or "429" in msg
                    or "502" in msg or "503" in msg or "504" in msg
                    or "overloaded" in lowered
                )
                # ── Auto-failover: one retry on the fast backup model ──────
                if retriable and self._switch_to_backup():
                    yield ErrorEvent(
                        message=f"⚠ {self._config.nvidia_model} unavailable "
                                f"({msg[:80]}) — switching to {self._model_label} and retrying…"
                    )
                    continue   # retry the loop iteration with the backup client

                if "401" in msg or "unauthorized" in lowered:
                    yield ErrorEvent(
                        message="NVIDIA NIM rejected the API key (401). "
                                "Update NVIDIA_KEYS in config.py or set NVIDIA_API_KEY env var."
                    )
                elif "404" in msg and "model" in lowered:
                    yield ErrorEvent(
                        message=f"Model '{self._model_label}' not available on NVIDIA NIM (404). "
                                "Run /models and pick another."
                    )
                elif retriable:
                    yield ErrorEvent(
                        message="NVIDIA NIM timed out / rate-limited even on failover model. "
                                "Wait a moment and try again."
                    )
                elif "connection" in lowered:
                    yield ErrorEvent(message=f"Cannot reach NVIDIA NIM: {msg[:200]}")
                else:
                    yield ErrorEvent(message=f"NVIDIA NIM error: {msg[:400]}")
                return

            # ── Parse response ─────────────────────────────────────────────────
            msg        = response.get("message", {})
            content    = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", []) or []

            log.debug("Response: tool_calls=%d  content_len=%d", len(tool_calls), len(content))

            # ── Empty-response retry ──────────────────────────────────────────
            # Some NIM models intermittently finish with no content AND no tool
            # calls (finish_reason='tool_calls' but an empty tool_calls array).
            # One cheap retry recovers those turns instead of showing a blank
            # answer.
            if not tool_calls and not content.strip() and _empty_retries < 2:
                _empty_retries += 1
                log.warning("Empty response — retrying (%d/2)", _empty_retries)
                iterations -= 1          # do not spend the tool-call budget on it
                continue

            # ── Handle tool calls ──────────────────────────────────────────────
            if tool_calls:
                # Add the assistant's message (with tool_calls) to history
                messages.append({
                    "role":       "assistant",
                    "content":    content,
                    "tool_calls": [
                        {
                            "id":   tc.get("id", "call_x"),
                            "type": "function",
                            "function": {
                                "name":      tc["function"]["name"],
                                # OpenAI wire format needs a JSON string
                                "arguments": json.dumps(tc["function"].get("arguments", {}) or {}),
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    name       = tc["function"]["name"]
                    params     = tc["function"].get("arguments", {}) or {}
                    tool_call_id = tc.get("id", f"call_{iterations}_{name}")

                    # ── Loop detection ─────────────────────────────────────────
                    _key = f"{name}:{list(params.values())[:1]}"
                    if _tool_history.count(_key) >= 2:
                        log.warning("Tool loop detected on %s — forcing final answer", name)
                        messages.append({
                            "role":    "user",
                            "content": (
                                "You have called the same tool multiple times. "
                                "Stop calling tools now. "
                                "Use the data you already have and give your final answer."
                            ),
                        })
                        break

                    _tool_history.append(_key)

                    yield ToolCallEvent(name=name, params=params, tool_call_id=tool_call_id)

                    result   = dispatch(name, params)
                    is_error = result.startswith('{"error"')

                    yield ToolResultEvent(name=name, result=result, is_error=is_error)

                    messages.append({
                        "role":    "tool",
                        "content": result,
                        "name":    name,
                    })

                continue  # Let the model process results

            # ── Final text answer ──────────────────────────────────────────────
            final_text = content.strip()

            if final_text:
                session.add("user",      user_message)
                session.add("assistant", final_text)
                yield FinalAnswerEvent(text=final_text)
                return

            if response.get("done"):
                yield FinalAnswerEvent(text="[No response generated. Try rephrasing your question.]")
                return

        # Iteration limit reached — try to extract any partial answer from messages
        partial = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content", "").strip():
                partial = m["content"].strip()
                break

        if partial:
            yield FinalAnswerEvent(
                text=partial + "\n\n*[Analysis based on data collected before iteration limit]*"
            )
        else:
            yield ErrorEvent(
                message=(
                    f"Reached {MAX_TOOL_ITERATIONS}-tool-call limit. "
                    "The model needs more iterations than allowed. "
                    "Try asking a more specific question (e.g. '/spot' instead of a full analysis), "
                    "or use '/models' to pick a different model."
                )
            )
