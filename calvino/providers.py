"""Provider abstraction over model APIs.

The harness depends only on the `Provider` protocol, so adding Google, OpenAI, or
local Ollama later means writing another class — no change to the harness. A
`GenerationRequest` carries the fixed, recorded knobs (temperature, thinking,
seed); a `GenerationResult` carries the output plus provider metadata that is
persisted verbatim for reproducibility.

These two are transient in-process DTOs — never persisted or parsed from untrusted
input (that boundary is the Pydantic `Generation` the harness builds from a
result), so they are plain dataclasses.

Phase 1 ships Anthropic + a deterministic Mock (for tests, and for wiring the
pipeline end-to-end without spending tokens).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .models import Condition


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model: str
    condition: Condition
    temperature: float
    seed: int | None = None
    max_tokens: int = 1024
    # Extended-thinking knobs; only used when condition == THINKING.
    #   effort: "high"/"medium"/"low" for the modern adaptive API (Opus 4.8+:
    #           thinking.type.adaptive + output_config.effort).
    #   thinking_budget: token budget for the LEGACY enabled API (older models that
    #           reject adaptive) — used only as the provider's fallback.
    effort: str = "high"
    thinking_budget: int = 2048
    # When set, force the provider to return JSON (judge calls). Eliminates the
    # plain-text JSON syntax failures (unescaped quotes, mis-balanced braces) that
    # the nested rubrics provoke. Each provider uses its native mechanism:
    # Anthropic forced tool-use (schema-conformant), OpenAI response_format,
    # Google JSON mime-type, Ollama format=json.
    json_schema: dict | None = None


@dataclass
class GenerationResult:
    output_text: str
    provider_meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


# Headroom for max_tokens when thinking is on, so reasoning + the small JSON output
# don't truncate even at high/xhigh/max effort. A ceiling only — billing is for tokens
# actually generated (matches the vendor's own adaptive-thinking example).
_THINKING_MAX_TOKENS = 16000


class ProviderError(RuntimeError):
    """Raised when a provider call fails after exhausting retries."""


class QuotaError(ProviderError):
    """Provider call failed because the API quota / rate limit is exhausted (HTTP 429,
    RESOURCE_EXHAUSTED, etc.). Distinct from ProviderError so callers can SKIP the rest
    of that provider's work and resume on a later run, rather than crash or hammer a
    dead quota through full backoff on every remaining call."""


# Substrings (lower-cased) that mark a quota / rate-limit failure across SDKs. Gemini
# raises RESOURCE_EXHAUSTED / 429; Anthropic & OpenAI raise 429 / rate-limit / quota.
_QUOTA_SIGNS = (
    "429", "resource_exhausted", "resource exhausted", "quota",
    "rate limit", "rate_limit", "ratelimit", "too many requests", "insufficient_quota",
)


def _is_quota_error(exc: BaseException) -> bool:
    return any(sign in str(exc).lower() for sign in _QUOTA_SIGNS)


def _json_safe(value):
    """Coerce a value to JSON-native types so it can live in provider_meta and
    survive Generation serialization. SDK usage objects can carry nested pydantic
    models that otherwise break model_dump_json; default=str stringifies anything
    left over."""
    return json.loads(json.dumps(value, default=str))


def _usage_dict(usage) -> dict:
    """Best-effort plain dict of token usage from an SDK usage object."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:  # noqa: BLE001
            pass
    return dict(getattr(usage, "__dict__", {}) or {})


def _with_retries(fn, *, attempts: int = 4, base_delay: float = 1.0, sleep=time.sleep):
    """Call `fn` with exponential backoff. `sleep` is injectable for tests.

    A transient rate-limit (429) is retried like any other failure — the backoff often
    clears it. But if the FINAL failure still looks like a quota/rate-limit, raise
    `QuotaError` (not plain `ProviderError`) so the caller can skip the rest of that
    provider's work and resume later instead of hammering a dead quota."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — provider exceptions vary by SDK
            last = exc
            if attempt == attempts - 1:
                break
            sleep(base_delay * (2**attempt))
    if _is_quota_error(last):
        raise QuotaError(f"provider quota/rate limit exhausted: {last}") from last
    raise ProviderError(f"provider call failed after {attempts} attempts") from last


class AnthropicProvider:
    """Anthropic Messages API provider.

    Modern adaptive thinking (Fable/Mythos 5+, Opus 4.8+: thinking.type.adaptive +
    output_config.effort) is the canonical thinking call and takes NO custom
    temperature — the API uses its default (1.0) and rejects e.g. temperature=0.7. So a
    THINKING condition omits temperature entirely; the requested value is still recorded
    in provider_meta (temperature_effective is then null = model default used). The
    NORMAL path sends no thinking param, accepts temp 0.7, and still thinks adaptively at
    the default effort — that is what the existing default-fable generations are. The
    LEGACY enabled+budget_tokens shape is only a fallback for OLDER models that reject
    adaptive; fable/mythos REJECT enabled ("use adaptive + output_config.effort"), so the
    fallback matches narrowly and never rewrites their calls. Thinking blocks are dropped
    from the returned text; only the final text blocks form the output.
    """

    def __init__(self, client=None):
        # Lazy import so the dependency isn't required unless this provider is used.
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        thinking_on = request.condition == Condition.THINKING
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if thinking_on:
            # Canonical adaptive-thinking call (Fable/Mythos 5+, Opus 4.8+): thinking.type.adaptive
            # + output_config.effort, and NO temperature (explicit thinking rejects a custom temp;
            # the API defaults to 1.0). Depth is set entirely by effort. An explicit effort ladder
            # is therefore temp-1.0 — the only varying knob is effort.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": request.effort}
            # Ample room so thinking + the (small) output never truncate (billed for actual use).
            kwargs["max_tokens"] = max(request.max_tokens, _THINKING_MAX_TOKENS)
        else:
            kwargs["temperature"] = request.temperature

        # Forced tool-use guarantees schema-conformant, syntactically valid JSON.
        if request.json_schema is not None:
            kwargs["tools"] = [{
                "name": "submit_scores",
                "description": "Return the rubric scores as structured JSON.",
                "input_schema": request.json_schema,
            }]
            # Extended thinking forbids forcing a named tool — only tool_choice "auto"
            # is permitted with thinking on. Without thinking we force the tool for a
            # guaranteed schema-conformant call; with thinking we let the model choose
            # (it reliably calls the tool, and extraction falls back to text if not).
            kwargs["tool_choice"] = (
                {"type": "auto"} if thinking_on else {"type": "tool", "name": "submit_scores"}
            )

        def _call():
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                # ONLY older models that genuinely lack adaptive thinking fall back to the legacy
                # enabled + budget_tokens shape. Match narrowly — "adaptive" AND an explicit
                # unsupported phrase — so a fable/mythos call is NEVER rewritten to enabled: those
                # models REJECT enabled ("thinking.type.enabled is not supported ... use adaptive +
                # output_config.effort"), which is exactly the 400 a broad match caused. (kwargs is
                # mutated, so a _with_retries re-call stays on the legacy shape.)
                unsupported = any(p in msg for p in
                                  ("not supported", "unsupported", "not valid", "unknown", "invalid"))
                if thinking_on and "adaptive" in msg and unsupported:
                    kwargs.pop("output_config", None)
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": request.thinking_budget}
                    kwargs["max_tokens"] = max(request.max_tokens, request.thinking_budget + 1024)
                    kwargs["temperature"] = 1.0  # legacy extended thinking requires temperature 1.0
                    return self._client.messages.create(**kwargs)
                # Newest models (e.g. Opus 4.8) deprecate/reject temperature (non-thinking path).
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature", None)
                    return self._client.messages.create(**kwargs)
                raise

        resp = _with_retries(_call)

        if request.json_schema is not None:
            # Extract the tool call's input dict as JSON text. Under extended thinking
            # (tool_choice auto) the model may answer in plain text instead — fall back
            # to the text blocks so the caller's JSON parser can still recover it.
            text = ""
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    text = json.dumps(block.input)
                    break
            if not text:
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                ).strip()
        else:
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            ).strip()

        meta = {
            "provider": "anthropic",
            "model": getattr(resp, "model", request.model),
            "thinking_enabled": thinking_on,
            # Adaptive-thinking effort actually requested (None when thinking off). Records the
            # ladder rung (low/high/xhigh/...) so a run is self-documenting for the analysis.
            "effort": request.effort if thinking_on else None,
            "requested_temperature": request.temperature,
            # EFFECTIVE temperature actually sent: extended thinking forces 1.0, else the
            # requested value (kwargs["temperature"] is set on both paths above).
            "temperature_effective": kwargs.get("temperature"),
            "stop_reason": getattr(resp, "stop_reason", None),
            "usage": _usage_dict(getattr(resp, "usage", None)),
        }
        return GenerationResult(output_text=text, provider_meta=_json_safe(meta))


class OpenAIProvider:
    """OpenAI Chat Completions provider. Reads OPENAI_API_KEY from the environment
    (never from config), so the key stays in the shell that launches the process.

    Kept generic (no forced response_format) so the same provider serves both
    generation and judging; the judge orchestrator already tolerates fenced/dirty
    JSON. Reasoning models (o-series, gpt-5.x) often reject a custom temperature —
    the call retries without it if the API complains.
    """

    def __init__(self, client=None):
        if client is None:
            import openai

            # timeout caps a single call so a stalled/rate-limited socket RAISES instead of
            # hanging (~600s SDK default x retries was a ~17min hang under concurrent same-key
            # load, 2026-07-02); max_retries=0 hands retrying to _with_retries (which also
            # converts a persistent 429 to QuotaError -> caller skips + resumes).
            client = openai.OpenAI(timeout=120.0, max_retries=0)  # reads OPENAI_API_KEY from env
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        thinking_on = request.condition == Condition.THINKING
        messages = [{"role": "user", "content": request.prompt}]

        kwargs: dict[str, Any] = {"model": request.model, "messages": messages}
        if thinking_on:
            # gpt-5.x reasoning knob. Reasoning models reject a custom temperature, so we
            # omit it (the temp-pop fallback below also covers any model that complains).
            kwargs["reasoning_effort"] = request.effort
        else:
            # NORMAL means "no thinking" — but gpt-5.x/o-series models REASON BY DEFAULT if
            # reasoning_effort is left unset, so we explicitly request reasoning OFF. gpt-5.5
            # supports {none,low,medium,high,xhigh} and REJECTS "minimal" (an o-series value), so
            # we use "none". Non-reasoning models
            # ignore the param; if any model rejects it, the fallback pops it and retries.
            kwargs["reasoning_effort"] = "none"
            kwargs["temperature"] = request.temperature
        if request.json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}  # guarantees valid JSON

        def _call():
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                # Reasoning models reject a non-default temperature, and some models reject
                # reasoning_effort entirely. Pop whichever the API complained about and retry;
                # popped values are recorded below (None => the model default was used).
                s = str(exc).lower()
                popped = False
                if "temperature" in s and "temperature" in kwargs:
                    kwargs.pop("temperature"); popped = True
                if "reasoning_effort" in s and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort"); popped = True
                if popped:
                    return self._client.chat.completions.create(**kwargs)
                raise

        resp = _with_retries(_call)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        meta = {
            "provider": "openai",
            "model": getattr(resp, "model", request.model),
            "thinking_enabled": thinking_on,
            # EFFECTIVE values actually sent (post-fallback): None => the API rejected it and
            # its own default was used. Record both so reasoning/temperature are auditable and
            # a "no-thinking" run is provably reasoning-minimal.
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "temperature_requested": request.temperature,
            "temperature_effective": kwargs.get("temperature"),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": _usage_dict(getattr(resp, "usage", None)),
        }
        return GenerationResult(output_text=text, provider_meta=_json_safe(meta))


class XAIProvider:
    """xAI Grok provider via the OpenAI SDK pointed at xAI's OpenAI-compatible
    endpoint (https://api.x.ai/v1). Reads XAI_API_KEY from the environment (never
    config), so the key stays in the shell that launches the process.

    Uses the Responses API (`client.responses.create`) — xAI's current interface for
    Grok 4.x. JSON is not force-formatted: the fixed scoring prompt already instructs
    "Output ONLY a JSON object", the orchestrator tolerates fenced/dirty JSON, and the
    raw reply is persisted before parsing — so prompt-level JSON is enough and we stay
    robust across SDK/endpoint format-param churn. Grok is a reasoning model and may
    reject a custom temperature; the call retries without it if the API complains.

    This is the 4th-vendor (xAI) neutral judge — a training lineage independent of
    Anthropic, OpenAI, and Google. It is the control that lifts the lineage-geometry
    finding from one same-lab pair to >=2 independent neutral lineages.
    """

    def __init__(self, client=None):
        if client is None:
            import os

            import openai

            client = openai.OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.environ.get("XAI_API_KEY"),
                # 600s (not 120s): grok-4.x reasons heavily by default and a single
                # comparison can exceed two minutes; a 120s cap timed out mid-reason,
                # exhausted retries, and raised — killing the judge (2026-07-09).
                timeout=600.0,      # cap hangs; see OpenAIProvider note
                max_retries=0,      # _with_retries owns retry + QuotaError conversion
            )
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # Judges score with thinking OFF and temperature 0 for determinism; the
        # request's json_schema is intentionally ignored (prompt-level JSON, above).
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": request.prompt,
            "temperature": request.temperature,
        }

        def _call():
            try:
                return self._client.responses.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                # Reasoning models often reject a non-default temperature.
                if "temperature" in str(exc).lower():
                    kwargs.pop("temperature", None)
                    return self._client.responses.create(**kwargs)
                raise

        resp = _with_retries(_call)

        text = (getattr(resp, "output_text", None) or "").strip()
        if not text:  # fallback: stitch text out of the structured output items
            parts: list[str] = []
            for item in getattr(resp, "output", None) or []:
                for chunk in getattr(item, "content", None) or []:
                    piece = getattr(chunk, "text", None)
                    if piece:
                        parts.append(piece)
            text = "".join(parts).strip()

        meta = {
            "provider": "xai",
            "model": getattr(resp, "model", request.model),
            "thinking_enabled": False,
            "requested_temperature": request.temperature,
            "usage": _usage_dict(getattr(resp, "usage", None)),
        }
        return GenerationResult(output_text=text, provider_meta=_json_safe(meta))


class OpenRouterProvider:
    """OpenRouter (https://openrouter.ai/api/v1) via the OpenAI SDK — a unified
    OpenAI-compatible gateway fronting many vendors. Reads OPENROUTER_API_KEY from the
    environment (never config). Model ids are vendor-prefixed slugs (e.g.
    'qwen/qwen3.7-max'); set the slug as the judge's `model`.

    Used here for a THIRD independent neutral lineage (qwen / Alibaba) — distinct from
    Anthropic, OpenAI, Google, and xAI — to break the fable-vs-gpt-5.5 lineage split.

    Like XAIProvider, JSON is not force-formatted: the fixed scoring / pairwise prompts
    already instruct JSON, the orchestrators tolerate fenced/dirty JSON, and raw replies
    are persisted — robust across the many models OpenRouter fronts (not all support
    response_format). Reasoning models may reject a custom temperature; the call retries
    without it. (Lineage independence is an assumption, not a guarantee — a model distilled
    from GPT/Claude would inherit their prior; position-screen and sanity-check first.)
    """

    def __init__(self, client=None):
        if client is None:
            import os

            import openai

            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ProviderError(
                    "OPENROUTER_API_KEY is not set in this environment. Export it in the "
                    "terminal before running (do NOT rely on an OPENAI_API_KEY fallback): "
                    "export OPENROUTER_API_KEY=sk-or-v1-..."
                )
            client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                                    timeout=120.0, max_retries=0)  # cap hangs; see OpenAIProvider note
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }

        def _call():
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if "temperature" in str(exc).lower():
                    kwargs.pop("temperature", None)
                    return self._client.chat.completions.create(**kwargs)
                raise

        resp = _with_retries(_call)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        meta = {
            "provider": "openrouter",
            "model": getattr(resp, "model", request.model),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": _usage_dict(getattr(resp, "usage", None)),
        }
        return GenerationResult(output_text=text, provider_meta=_json_safe(meta))


class GoogleProvider:
    """Google Gemini provider via the google-genai SDK. Reads GEMINI_API_KEY /
    GOOGLE_API_KEY from the environment (never config), so the key stays in the
    shell that launches the process.

    Like the OpenAI provider: generic (no forced response schema), and it retries
    without temperature if a reasoning model rejects it. Gemini's `.text` returns
    the final answer even when the model thinks internally; if it comes back empty
    (e.g. safety block) the candidate parts are joined as a fallback.
    """

    def __init__(self, client=None, timeout_ms: int = 120_000):
        if client is None:
            from google import genai

            # 120s per-request timeout (HttpOptions.timeout is in MILLISECONDS). Without it
            # the client has NO timeout: a rate-limited or stalled socket blocks forever
            # (observed 2026-07-02 — a gemini call hung >3min under concurrent same-key load
            # while _with_retries never fired, because a silent hang raises no exception).
            # A timeout forces the SDK to RAISE, so _with_retries can back off and retry, and
            # a genuinely dead quota surfaces as QuotaError so the caller skips + resumes.
            client = genai.Client(  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env
                http_options={"timeout": timeout_ms},
            )
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        config: dict[str, Any] = {"temperature": request.temperature}
        if request.json_schema is not None:
            config["response_mime_type"] = "application/json"  # guarantees valid JSON

        def _call():
            try:
                return self._client.models.generate_content(
                    model=request.model, contents=request.prompt, config=config,
                )
            except Exception as exc:  # noqa: BLE001
                if "temperature" in str(exc).lower():
                    return self._client.models.generate_content(
                        model=request.model, contents=request.prompt,
                    )
                raise

        resp = _with_retries(_call)
        text = (getattr(resp, "text", None) or "").strip()
        if not text:  # fallback: stitch candidate parts if .text is empty
            for cand in getattr(resp, "candidates", None) or []:
                parts = getattr(getattr(cand, "content", None), "parts", None) or []
                text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
                if text:
                    break

        finish = None
        cands = getattr(resp, "candidates", None)
        if cands:
            finish = getattr(cands[0], "finish_reason", None)
        meta = {
            "provider": "google",
            "model": request.model,
            "finish_reason": finish,
            "usage": _usage_dict(getattr(resp, "usage_metadata", None)),
        }
        return GenerationResult(output_text=text, provider_meta=_json_safe(meta))


class OllamaProvider:
    """Local Ollama provider via the /api/chat endpoint. Free to run, and unlike
    the Anthropic API it honours `seed` (recorded for reproducibility).

    Ollama does not expose extended thinking uniformly, so the thinking condition
    is treated as normal here and flagged in provider_meta; keep the reasoning-
    depth comparison on a provider that genuinely supports the toggle.
    """

    # def __init__(self, host: str = "http://localhost:11434", client=None, timeout: float = 600.0,
    def __init__(self, host: str = "http://127.0.0.1:11434", client=None, timeout: float = 600.0,
                 think: bool = False, num_ctx: int = 8192):
        self._host = host.rstrip("/")
        # Reasoning models (e.g. gemma4) emit a separate `thinking` field; with
        # thinking ON the reasoning can consume the whole output budget and leave
        # `content` empty. We disable it by default (Ollama runs are flattened to
        # "normal" anyway) and give context headroom so the JSON answer fits.
        self._think = think
        self._num_ctx = num_ctx
        # `client` is any object with .post(url, json=...) -> response (.json(),
        # .raise_for_status()); injectable for tests. Defaults to an httpx client.
        # Large local models (e.g. 26B) can take minutes to cold-load + generate,
        # so the default read timeout is generous.
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout)
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        options: dict[str, Any] = {"temperature": request.temperature, "num_ctx": self._num_ctx}
        if request.seed is not None:
            options["seed"] = request.seed
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "think": self._think,
            "options": options,
        }
        if request.json_schema is not None:
            payload["format"] = "json"  # Ollama: constrain output to valid JSON

        def _call():
            resp = self._client.post(f"{self._host}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

        data = _with_retries(_call)
        text = data.get("message", {}).get("content", "").strip()
        meta = {
            "provider": "ollama",
            "model": data.get("model", request.model),
            "thinking_enabled": False,  # not supported here; condition flattened
            "requested_condition": request.condition.value,
            "seed": request.seed,
            "eval_count": data.get("eval_count"),
            "done_reason": data.get("done_reason"),
        }
        return GenerationResult(output_text=text, provider_meta=meta)


class MockProvider:
    """Deterministic provider for tests and dry runs — no network call.

    Dual-role: for a generation prompt it returns a canned 4-sentence paragraph;
    for a scoring prompt (detected by the JSON-output marker) it returns a valid
    judge JSON payload whose evidence spans are verbatim in the canned paragraph,
    so the whole pipeline runs on mocks. A per-prompt `responses` map overrides
    either behaviour."""

    DEFAULT = (
        "The smuggler's lantern burned steady on the headland as the patrol passed. "
        "He raised it twice, and far offshore the ship began to move. "
        "At dawn the widow carried the same lantern down into the flooded cellar. "
        "Its light slid over the waterline, naming each thing the tide had taken."
    )
    # A span guaranteed to appear verbatim in DEFAULT, used for every dimension.
    _JUDGE_SPAN = "the same lantern"
    _SCORING_MARKER = "Output ONLY a JSON object"

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None):
        self._responses = responses or {}
        self._default = default if default is not None else self.DEFAULT
        self.calls: list[GenerationRequest] = []

    def _judge_json(self) -> str:
        import json

        dims = ("distinctness", "bridge", "tonal", "originality", "combinatorial")
        return json.dumps({  # FLAT shape, matching JUDGE_JSON_SCHEMA
            **{f"{d}_score": 4 for d in dims},
            **{f"{d}_evidence": self._JUDGE_SPAN for d in dims},
        })

    def _grid_judge_json(self) -> str:
        import json

        cell = lambda s: {"score": s, "evidence": self._JUDGE_SPAN}  # noqa: E731
        return json.dumps({
            "crossing_x": {"distinctness": cell(4), "bridge": cell(4)},
            "crossing_y": {"distinctness": cell(4), "bridge": cell(4)},
            "combinatorial_integrity": cell(3),
            "tonal": cell(4),
            "originality": cell(3),
        })

    # --- tableau (construct two) mock behaviours -------------------------------
    # A fixed phrase every mock tableau story contains, reused as the judge's
    # evidence span so span verification passes end-to-end on mocks.
    _TABLEAU_SPAN = "lay on the table"

    def _tableau_output(self, request: GenerationRequest) -> str:
        """Build a gate-COMPLIANT tableau from the prompt's own path lines
        ("R1: name -> name -> name"): six labelled stories, exactly 4 sentences,
        assigned cards in path order, no other cards. Sentence 4 varies
        deterministically with the model name so different models produce
        different-length tableaux (gives the mock forced choice something to
        decide) while two samples of one model stay identical (so same-model
        pairs abstain, exercising the §6 re-screen statistic)."""
        import hashlib
        import re as _re

        paths = _re.findall(r"(?m)^(R[123]|C[123]):\s*(.+)$", request.prompt)
        if len(paths) < 6:
            return self._default
        pad = int(hashlib.sha256(request.model.encode()).hexdigest(), 16) % 4
        tail = "The teller paused" + (", nodding slowly" * pad) + ", and the table was quiet."
        stories = []
        for label, line in paths[:6]:
            names = [n.strip() for n in line.split("->")]
            stories.append(
                f"{label}: {names[0]} lay on the table first. "
                f"Beside it {names[1]} waited its turn. "
                f"Then {names[2]} ended the line. {tail}"
            )
        return "\n".join(stories)

    def _tableau_judge_json(self, request: GenerationRequest) -> str:
        """Fill the item-specific flat field inventory, preferably straight from
        the request's json_schema (kind-aware: 0-5 scores, 0-1 checks, strings);
        evidence is the fixed phrase present in every mock story, so span
        verification passes end-to-end on mocks. Falls back to scraping `*_score`
        keys from the prompt when no schema is attached."""
        import json
        import re as _re

        payload: dict[str, object] = {}
        schema = request.json_schema or {}
        props = schema.get("properties") or {}
        if props:
            for name, spec in props.items():
                if spec.get("type") == "string":
                    payload[name] = self._TABLEAU_SPAN
                elif spec.get("maximum") == 1:
                    payload[name] = 1
                else:
                    payload[name] = 3 if name.startswith("gestalt") else 4
        else:
            keys = sorted(set(_re.findall(r"\b([A-Za-z0-9_]+)_score\b", request.prompt)))
            for k in keys:
                payload[f"{k}_score"] = 3 if k in ("coherence", "gestalt") else 4
                payload[f"{k}_evidence"] = self._TABLEAU_SPAN
        return json.dumps(payload)

    def _tableau_vote_json(self, request: GenerationRequest) -> str:
        """Position-independent mock vote: the longer tableau wins; equal lengths
        abstain. Stable across both presentation orders, so cross-model
        comparisons decide and same-model comparisons abstain."""
        import re as _re

        blocks = _re.findall(r'"""\n(.*?)\n"""', request.prompt, _re.DOTALL)
        if len(blocks) >= 2:
            a, b = blocks[-2], blocks[-1]
            if len(a) > len(b):
                w = "A"
            elif len(b) > len(a):
                w = "B"
            else:
                w = "abstain"
        else:
            w = "abstain"
        return json.dumps({"winner": w, "reason": "mock length rule"})

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if request.prompt in self._responses:
            text = self._responses[request.prompt]
        elif "crossing_x" in request.prompt:        # grid scoring prompt (more specific)
            text = self._grid_judge_json()
        elif "Tableau A:" in request.prompt and '"winner"' in request.prompt:
            text = self._tableau_vote_json(request)  # tableau forced choice
        elif '"winner"' in request.prompt:           # pairwise forced-choice prompt
            text = json.dumps({"winner": "A", "reason": "mock"})
        elif "gestalt_score" in request.prompt or "coherence_score" in request.prompt:
            text = self._tableau_judge_json(request)  # tableau scoring prompt (r2 / legacy r1)
        elif self._SCORING_MARKER in request.prompt:  # single-crossing scoring prompt
            text = self._judge_json()
        elif "3x3" in request.prompt and "R1:" in request.prompt:  # tableau generation
            text = self._tableau_output(request)
        else:
            text = self._default
        return GenerationResult(
            output_text=text,
            provider_meta={"provider": "mock", "model": request.model},
        )


def get_provider(name: str) -> Provider:
    """Resolve a provider by config name. 'mock' needs no API key and is used for
    dry runs and tests; 'anthropic' reads ANTHROPIC_API_KEY from the environment."""
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name in ("xai", "grok"):
        return XAIProvider()
    if name == "openrouter":
        return OpenRouterProvider()
    if name in ("google", "gemini"):
        return GoogleProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(
        f"unknown provider {name!r} (expected 'anthropic', 'openai', 'xai', 'openrouter', "
        "'google', 'ollama', or 'mock')"
    )
