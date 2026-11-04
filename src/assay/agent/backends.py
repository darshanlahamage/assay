"""Model backends: the API track (Gemini) and the local track (MLX).

A backend turns (system prompt, chat messages) into one reply. Local backends also
return per-token logprobs and entropies, which the API cannot give.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    # Local track only: {"prompt_last" | "action" | "gen_mean": array [n_layers, hidden]}
    activations: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str
    model: str

    def complete(self, system: str, messages: list[dict[str, str]]) -> Completion: ...


class QuotaExhausted(RuntimeError):
    """The API quota won't recover within this run; retrying only burns time."""


def _is_daily_quota(e: Exception) -> bool:
    text = str(e)
    return "PerDay" in text or "per day" in text.lower()


class GeminiBackend:
    name = "gemini"

    def __init__(self, model: str, temperature: float = 0.7, max_output_tokens: int = 8192,
                 max_retries: int = 6) -> None:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
        self._client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.quota_wait_max_s = quota_wait_max_s

    def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        from google.genai import errors, types

        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        import httpx

        for attempt in range(self.max_retries):
            try:
                r = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config)
                break
            except errors.APIError as e:
                if e.code == 429 and _is_daily_quota(e):
                    raise QuotaExhausted(f"{self.model}: daily quota exhausted ({e.message})") from e
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries - 1:
                    raise
            except (httpx.TransportError, ConnectionError):
                if attempt == self.max_retries - 1:
                    raise
            time.sleep(min(60, 2 ** attempt * 2))
        u = r.usage_metadata
        out_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0) if u else 0
        return Completion(text=r.text or "", input_tokens=(u.prompt_token_count or 0) if u else 0,
                          output_tokens=out_tokens)


class GroqBackend:
    """API track via Groq (OpenAI-compatible chat API), e.g. openai/gpt-oss-120b.

    Groq's free tier allows ~8000 tokens/minute and, per model, a daily token budget
    (200k for gpt-oss-120b: roughly 4-8 rollouts). Long conversations are kept under
    `max_prompt_chars` by shortening the oldest command outputs first (the task message
    and recent turns stay whole). With `quota_wait_max_s` > 0, hitting the daily budget
    pauses until Groq says tokens are available again (the budget is a rolling window),
    instead of giving up; sweeps use this to run unattended over several days.
    """

    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, temperature: float = 0.7, max_output_tokens: int = 4096,
                 max_prompt_chars: int = 18000, max_retries: int = 8,
                 quota_wait_max_s: float = 0) -> None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
        self._headers = {"Authorization": f"Bearer {key}"}
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_prompt_chars = max_prompt_chars
        self.max_retries = max_retries

    def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        import httpx

        budget = self.max_prompt_chars
        bad_calls = 0
        attempt = -1
        while attempt < self.max_retries - 1:
            attempt += 1
            msgs = [{"role": "system", "content": system},
                    *_as_tool_history(_compact(messages, budget))]
            body = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                    "max_completion_tokens": self.max_output_tokens,
                    "tools": _TOOLS, "tool_choice": "auto"}
            try:
                r = httpx.post(self.URL, headers=self._headers, json=body, timeout=180)
            except httpx.TransportError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage") or {}
                msg = d["choices"][0]["message"]
                return Completion(text=_tool_calls_to_text(msg),
                                  input_tokens=u.get("prompt_tokens", 0),
                                  output_tokens=u.get("completion_tokens", 0))
            text = r.text
            if r.status_code == 400 and "tool_use_failed" in text:
                # The model produced a malformed tool call. Recover the command from the raw
                # generation Groq returns; failing that, re-ask; failing that, report it back
                # to the model like a real harness (the loop shows a format error).
                salvaged = _salvage_tool_call(text)
                if salvaged:
                    return Completion(text=salvaged)
                bad_calls += 1
                if bad_calls <= 2:
                    continue
                return Completion(text="(invalid tool call: arguments were not valid JSON. "
                                       "Use the bash tool with a plain command string.)")
            if r.status_code == 413 or "too large" in text.lower():
                budget = int(budget * 0.7)      # request bigger than the per-minute budget
                continue
            if r.status_code == 429:
                if _is_daily_quota(text) or "tokens per day" in text.lower() or "(TPD)" in text \
                        or "(RPD)" in text:
                    wait = _retry_in(text)
                    if wait is not None and wait <= self.quota_wait_max_s:
                        print(f"      [groq daily budget used up; waiting {wait / 60:.0f} min]",
                              flush=True)
                        time.sleep(wait + 20)
                        attempt -= 1            # waiting for quota is not a failed attempt
                        continue
                    raise QuotaExhausted(f"{self.model}: daily quota exhausted ({text[:300]})")
                wait = float(r.headers.get("retry-after", 0) or 0)
                time.sleep(min(90, max(wait, 2 ** attempt)))
                continue
            if r.status_code >= 500 and attempt < self.max_retries - 1:
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            raise RuntimeError(f"Groq API error {r.status_code}: {text[:500]}")
        raise RuntimeError("Groq API: retries exhausted")


def _retry_in(text: str) -> float | None:
    """Seconds from Groq's "Please try again in 1h2m3.4s" (None if absent)."""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", text)
    if not m or not m.group(1):
        return None
    parts = {unit: v for v, unit in re.findall(r"([\d.]+)([hms])", m.group(1))}
    return (float(parts.get("h", 0)) * 3600 + float(parts.get("m", 0)) * 60
            + float(parts.get("s", 0)))


# Native tools for models trained on tool calling (gpt-oss). Calls are converted to the
# same text protocol the other backends use, so the agent loop is identical.
_TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in /repo inside the Linux container. "
                       "Returns the exit code and output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Finish the task with a final report of what you changed and the "
                       "state of the tests.",
        "parameters": {"type": "object", "properties": {"report": {"type": "string"}},
                       "required": ["report"]}}},
]


def _tool_calls_to_text(msg: dict) -> str:
    import json

    content = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    if not calls:
        return content
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    name = fn.get("name", "")
    if name == "submit" or "report" in args:
        return f"{content}\n```submit\n{args.get('report', '')}\n```"
    cmd = args.get("command", args.get("cmd", ""))
    if isinstance(cmd, list):               # e.g. ["bash", "-lc", "ls"]
        cmd = cmd[-1] if cmd[:2] in (["bash", "-lc"], ["bash", "-c"]) else " ".join(cmd)
    note = "" if len(calls) == 1 else "\n(only the first tool call per turn is run)"
    return f"{content}{note}\n```bash\n{cmd}\n```"


def _salvage_tool_call(error_body: str) -> str | None:
    """Recover a tool call from Groq's `failed_generation` when its JSON was invalid."""
    import json
    import re

    try:
        gen = json.loads(error_body)["error"]["failed_generation"]
    except (ValueError, KeyError, TypeError):
        return None
    try:
        call = json.loads(gen)
        args = call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return _tool_calls_to_text({"content": "", "tool_calls": [
            {"function": {"name": call.get("name", "bash"), "arguments": json.dumps(args)}}]})
    except (ValueError, AttributeError):
        pass
    # Unescaped quotes/newlines inside the command: take everything between the first
    # `"command": "` and the closing `"}`.
    m = re.search(r'"(command|cmd|report)"\s*:\s*"(.*)"\s*\}\s*\}?\s*$', gen, re.DOTALL)
    if not m:
        return None
    value = m.group(2).replace('\\n', '\n').replace('\\"', '"').replace("\\t", "\t")
    kind = "submit" if m.group(1) == "report" else "bash"
    return f"```{kind}\n{value}\n```"


def _as_tool_history(messages: list[dict[str, str]]) -> list[dict]:
    """Rewrite past actions as native tool calls and their outputs as tool results.

    Tool-calling models (gpt-oss) otherwise see their own actions as text blocks and start
    imitating that text format instead of calling tools.
    """
    import json

    from assay.agent import protocol

    out: list[dict] = []
    pending: str | None = None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            a = protocol.parse(m["content"])
            if a.kind in ("bash", "submit"):
                call_id = f"call_{i}"
                name, args = ("bash", {"command": a.content}) if a.kind == "bash" \
                    else ("submit", {"report": a.content})
                out.append({"role": "assistant", "content": a.reasoning or None,
                            "tool_calls": [{"id": call_id, "type": "function", "function": {
                                "name": name, "arguments": json.dumps(args)}}]})
                pending = call_id
                continue
            out.append({"role": "assistant", "content": m["content"]})
        elif pending is not None:
            out.append({"role": "tool", "tool_call_id": pending, "content": m["content"]})
            pending = None
            continue
        else:
            out.append({"role": m["role"], "content": m["content"]})
        pending = None
    return out


def _compact(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Shorten the oldest tool outputs until the conversation fits in `max_chars`.

    Keeps the first message (the task) and the last four messages whole.
    """
    msgs = [dict(m) for m in messages]
    total = sum(len(m["content"]) for m in msgs)
    for m in msgs[1:-4]:
        if total <= max_chars:
            break
        if m["role"] == "user" and len(m["content"]) > 400:
            short = m["content"][:300] + "\n... [older output shortened to save space]"
            total -= len(m["content"]) - len(short)
            m["content"] = short
    return msgs


class MLXBackend:
    name = "mlx"

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096,
                 enable_thinking: bool = False, adapter_path: str | None = None,
                 activation_layers: tuple[int, ...] = (), max_prompt_chars: int = 80000) -> None:
        from assay.inference.engine import InferenceEngine

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.max_prompt_chars = max_prompt_chars
        self.engine = InferenceEngine(model, adapter_path=adapter_path,
                                      activation_layers=tuple(activation_layers))

    def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        # Same shortening as the Groq track, so long rollouts fit the context window.
        messages = _compact(messages, self.max_prompt_chars)
        prompt = self.engine.format_chat(
            [{"role": "system", "content": system}, *messages],
            enable_thinking=self.enable_thinking,
        )
        out = self.engine.generate(prompt, max_tokens=self.max_tokens,
                                   temperature=self.temperature)
        return Completion(text=out.text, input_tokens=out.prompt_tokens,
                          output_tokens=len(out.token_ids),
                          logprobs=out.logprobs, entropies=out.entropies,
                          activations=out.activations)


class ScriptedBackend:
    """Replays fixed replies from a scenario file: a stand-in agent for testing the harness.

    Scenario YAML: {name: ..., description: ..., replies: [<reply text>, ...]}. Each reply
    uses the normal protocol (one ```bash or ```submit block). If the script runs out, it
    submits.
    """

    name = "script"

    def __init__(self, scenario_path: str) -> None:
        import yaml

        s = yaml.safe_load(Path(scenario_path).read_text())
        self.model = f"script:{s['name']}"
        self.description = s.get("description", "")
        self._replies = list(s["replies"])

    def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        if self._replies:
            return Completion(text=self._replies.pop(0))
        return Completion(text="```submit\n(script finished)\n```")


def make_backend(name: str, cfg: dict) -> Backend:
    c = cfg["backends"][name]
    if name == "gemini":
        return GeminiBackend(c["model"], c.get("temperature", 0.7),
                             c.get("max_output_tokens", 8192))
    if name == "mlx":
        return MLXBackend(c["model"], c.get("temperature", 0.7), c.get("max_tokens", 4096),
                          c.get("enable_thinking", False), c.get("adapter_path"),
                          tuple(c.get("activation_layers") or ()),
                          c.get("max_prompt_chars", 80000))
    if name == "groq":
        return GroqBackend(c["model"], c.get("temperature", 0.7), c.get("max_output_tokens", 4096),
                           c.get("max_prompt_chars", 18000),
                           quota_wait_max_s=c.get("quota_wait_max_s", 0))
    if name == "script":
        return ScriptedBackend(c["scenario"])
    raise ValueError(f"unknown backend {name!r}")
