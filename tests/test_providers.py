"""Provider tests that don't need network or API keys: the OllamaProvider request
build + response parsing (via a fake client), and get_provider resolution."""

import pytest

from calvino.models import Condition
from calvino.providers import (
    GenerationRequest,
    GoogleProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    QuotaError,
    _is_quota_error,
    _with_retries,
    get_provider,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeClient:
    """Records the last POST and returns a canned Ollama-shaped payload."""

    def __init__(self, payload):
        self._payload = payload
        self.last_url = None
        self.last_json = None

    def post(self, url, json=None):
        self.last_url = url
        self.last_json = json
        return FakeResponse(self._payload)


def _req(**kw):
    base = dict(prompt="write 4 sentences", model="llama3", condition=Condition.NORMAL, temperature=0.7)
    base.update(kw)
    return GenerationRequest(**base)


def test_ollama_builds_request_and_parses_response():
    client = FakeClient({"model": "llama3", "message": {"content": "  Four sentences here.  "},
                         "eval_count": 42, "done_reason": "stop"})
    provider = OllamaProvider(host="http://localhost:11434", client=client)

    result = provider.generate(_req(seed=7))

    assert client.last_url == "http://localhost:11434/api/chat"
    assert client.last_json["model"] == "llama3"
    assert client.last_json["stream"] is False
    assert client.last_json["options"]["temperature"] == 0.7
    assert client.last_json["options"]["seed"] == 7  # seed honoured
    assert result.output_text == "Four sentences here."  # stripped
    assert result.provider_meta["provider"] == "ollama"
    assert result.provider_meta["eval_count"] == 42


def test_ollama_disables_thinking_by_default():
    # Reasoning models route output through a `thinking` field that can starve
    # `content`; the provider disables it by default and sets a context budget.
    client = FakeClient({"message": {"content": "x"}})
    OllamaProvider(client=client).generate(_req())
    assert client.last_json["think"] is False
    assert client.last_json["options"]["num_ctx"] >= 4096


def test_ollama_omits_seed_when_none():
    client = FakeClient({"message": {"content": "x"}})
    OllamaProvider(client=client).generate(_req(seed=None))
    assert "seed" not in client.last_json["options"]


def test_ollama_flattens_thinking_condition():
    client = FakeClient({"message": {"content": "x"}})
    result = OllamaProvider(client=client).generate(_req(condition=Condition.THINKING))
    assert result.provider_meta["thinking_enabled"] is False
    assert result.provider_meta["requested_condition"] == "thinking"


def test_ollama_retries_then_raises_on_persistent_failure(monkeypatch):
    monkeypatch.setattr("calvino.providers.time.sleep", lambda *_: None)  # no real backoff wait

    calls = {"n": 0}

    class BoomClient:
        def post(self, url, json=None):
            calls["n"] += 1
            raise RuntimeError("connection refused")

    provider = OllamaProvider(client=BoomClient())
    with pytest.raises(ProviderError):
        provider.generate(_req())
    assert calls["n"] == 4  # retried the full budget before giving up


def test_is_quota_error_matches_429_and_resource_exhausted():
    assert _is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert _is_quota_error(Exception("Rate limit reached for requests"))
    assert _is_quota_error(Exception("insufficient_quota"))
    assert not _is_quota_error(RuntimeError("connection refused"))
    assert not _is_quota_error(ValueError("bad request 400"))


def test_with_retries_raises_quotaerror_when_final_failure_is_quota():
    # A persistent quota/rate-limit failure surfaces as QuotaError (a ProviderError
    # subclass) so callers can skip-and-resume instead of crashing.
    def boom():
        raise RuntimeError("429 Too Many Requests")
    with pytest.raises(QuotaError):
        _with_retries(boom, attempts=2, sleep=lambda *_: None)
    # QuotaError IS-A ProviderError, so existing `except ProviderError` still catches it.
    assert issubclass(QuotaError, ProviderError)


def test_with_retries_plain_provider_error_for_non_quota():
    def boom():
        raise RuntimeError("connection refused")
    with pytest.raises(ProviderError) as ei:
        _with_retries(boom, attempts=2, sleep=lambda *_: None)
    assert not isinstance(ei.value, QuotaError)


class _FakeOpenAIClient:
    """Mimics the openai SDK shape: client.chat.completions.create(...) -> object
    with .choices[0].message.content and .model/.usage."""

    def __init__(self, content):
        self._content = content
        self.last_kwargs = None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs
                msg = type("M", (), {"content": outer._content})()
                choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
                return type("R", (), {"choices": [choice], "model": kwargs["model"], "usage": None})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_openai_builds_request_and_parses_response():
    client = _FakeOpenAIClient('{"distinctness": {"score": 4, "evidence": "x"}}')
    provider = OpenAIProvider(client=client)
    result = provider.generate(_req(model="gpt-4.1", temperature=0.0))
    assert client.last_kwargs["model"] == "gpt-4.1"
    assert client.last_kwargs["temperature"] == 0.0
    assert client.last_kwargs["messages"][0]["content"] == "write 4 sentences"
    assert result.output_text.startswith("{")
    assert result.provider_meta["provider"] == "openai"


def test_openai_retries_without_temperature_when_rejected():
    # Reasoning models (gpt-5.x) reject a custom temperature; provider must retry.
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported value: 'temperature' is not supported")
            msg = type("M", (), {"content": "{}"})()
            choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
            return type("R", (), {"choices": [choice], "model": kwargs["model"], "usage": None})()

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()
    result = OpenAIProvider(client=client).generate(_req(model="gpt-5.5", temperature=0.7))
    assert len(calls) == 2  # first with temperature (fails), second without (ok)
    assert "temperature" not in calls[1]
    assert result.output_text == "{}"


def test_openai_normal_requests_no_reasoning_and_records_effective_temp():
    # NORMAL ("no thinking") must explicitly request reasoning OFF so gpt-5.x models don't
    # reason by default. gpt-5.5 rejects "minimal" and supports "none", so we send "none", and
    # record the effective temperature/reasoning for auditability.
    client = _FakeOpenAIClient("hi")
    result = OpenAIProvider(client=client).generate(_req(model="gpt-5.5", temperature=0.7))
    assert client.last_kwargs["reasoning_effort"] == "none"
    assert client.last_kwargs["temperature"] == 0.7
    assert result.provider_meta["reasoning_effort"] == "none"
    assert result.provider_meta["temperature_requested"] == 0.7
    assert result.provider_meta["temperature_effective"] == 0.7
    assert result.provider_meta["thinking_enabled"] is False


def test_openai_normal_pops_temperature_but_keeps_no_reasoning():
    # If a reasoning model rejects the custom temperature, we pop it (effective temp -> None,
    # meaning the model default was used) but KEEP reasoning_effort="none".
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported value: 'temperature' is not supported")
            msg = type("M", (), {"content": "{}"})()
            choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
            return type("R", (), {"choices": [choice], "model": kwargs["model"], "usage": None})()

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()
    result = OpenAIProvider(client=client).generate(_req(model="gpt-5.5", temperature=0.7))
    assert len(calls) == 2
    assert calls[1]["reasoning_effort"] == "none"  # reasoning knob survives the temp pop
    assert "temperature" not in calls[1]
    assert result.provider_meta["temperature_effective"] is None  # model default used
    assert result.provider_meta["reasoning_effort"] == "none"


def test_openai_thinking_sets_reasoning_effort_and_omits_temperature():
    # A thinking judge (THINKING condition) must send reasoning_effort and NOT temperature
    # (reasoning models reject a custom temperature).
    from calvino.models import Condition
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            msg = type("M", (), {"content": '{"x":1}'})()
            choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
            return type("R", (), {"choices": [choice], "model": kwargs["model"], "usage": None})()

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()
    result = OpenAIProvider(client=client).generate(
        _req(model="gpt-5.5", condition=Condition.THINKING, effort="high")
    )
    assert len(calls) == 1                         # no temperature round-trip needed
    assert calls[0]["reasoning_effort"] == "high"
    assert "temperature" not in calls[0]
    assert result.provider_meta["reasoning_effort"] == "high"
    assert result.provider_meta["thinking_enabled"] is True


def test_anthropic_retries_without_temperature_when_deprecated():
    # Newest models (Opus 4.8) reject `temperature`; provider must drop it and retry.
    from calvino.providers import AnthropicProvider

    calls = []
    block = type("B", (), {"type": "text", "text": "ok"})()
    resp = type("R", (), {"content": [block], "model": "claude-opus-4-8",
                          "stop_reason": "end_turn", "usage": None})()

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("`temperature` is deprecated for this model.")
            return resp

    client = type("Cl", (), {"messages": _Messages()})()
    result = AnthropicProvider(client=client).generate(_req(model="claude-opus-4-8", temperature=0.7))
    assert len(calls) == 2 and "temperature" not in calls[1]
    assert result.output_text == "ok"


def test_anthropic_meta_is_json_safe_for_generation():
    # Regression: SDK usage objects can carry nested non-serializable bits; the
    # provider must sanitize provider_meta so the Generation it feeds serializes.
    from calvino.providers import AnthropicProvider
    from calvino.models import Generation, ObjectType, Condition as Cond, PromptVariant

    class _Weird:  # not JSON-serializable, no model_dump
        def __init__(self):
            self.input_tokens = 5
            self.nested = object()

    block = type("B", (), {"type": "text", "text": "a paragraph"})()
    resp = type("R", (), {"content": [block], "model": "claude-x",
                          "stop_reason": "end_turn", "usage": _Weird()})()
    client = type("Cl", (), {"messages": type("M", (), {"create": lambda self, **k: resp})()})()

    result = AnthropicProvider(client=client).generate(_req(model="claude-x"))
    import json as _json
    _json.dumps(result.provider_meta)  # must not raise

    gen = Generation(
        id="g", item_id="i", object="o", object_type=ObjectType.TAROT, model="claude-x",
        condition=Cond.NORMAL, prompt_variant=PromptVariant.BASE, sample_idx=0,
        temperature=0.7, rendered_prompt="p", output_text=result.output_text,
        provider_meta=result.provider_meta,
    )
    assert gen.model_dump_json()  # the path that crashed before the fix


class _FakeGenAIClient:
    """Mimics google-genai: client.models.generate_content(model, contents, config)
    -> object with .text / .candidates / .usage_metadata. `fail_on_temp` simulates
    a reasoning model rejecting a temperature config."""

    def __init__(self, text, *, fail_on_temp=False):
        self._text = text
        self._fail_on_temp = fail_on_temp
        self.calls = []
        outer = self

        class _Models:
            def generate_content(self, **kwargs):
                outer.calls.append(kwargs)
                cfg = kwargs.get("config") or {}
                if outer._fail_on_temp and "temperature" in cfg:
                    raise RuntimeError("temperature is not supported for this model")
                cand = type("C", (), {"finish_reason": "STOP", "content": None})()
                return type("R", (), {"text": outer._text, "candidates": [cand],
                                      "usage_metadata": None})()

        self.models = _Models()


def test_google_builds_request_and_parses_response():
    client = _FakeGenAIClient('{"distinctness": {"score": 4, "evidence": "x"}}')
    result = GoogleProvider(client=client).generate(_req(model="gemini-3.5", temperature=0.7))
    assert client.calls[0]["model"] == "gemini-3.5"
    assert client.calls[0]["contents"] == "write 4 sentences"
    assert client.calls[0]["config"]["temperature"] == 0.7
    assert result.output_text.startswith("{")
    assert result.provider_meta["provider"] == "google"


def test_google_retries_without_temperature_when_rejected():
    client = _FakeGenAIClient("ok", fail_on_temp=True)
    result = GoogleProvider(client=client).generate(_req(model="gemini-3.5", temperature=0.7))
    assert len(client.calls) == 2
    assert "config" not in client.calls[1] or "temperature" not in (client.calls[1].get("config") or {})
    assert result.output_text == "ok"


def test_anthropic_json_schema_uses_forced_tool_use():
    # With a json_schema, the provider forces tool-use and returns the tool input
    # as JSON text (guaranteeing valid, schema-conformant JSON).
    from calvino.providers import AnthropicProvider
    import json as _json

    captured = {}
    tool_block = type("B", (), {"type": "tool_use", "input": {"distinctness": {"score": 4}}})()
    resp = type("R", (), {"content": [tool_block], "model": "claude-x", "stop_reason": "tool_use", "usage": None})()

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return resp

    client = type("Cl", (), {"messages": _Messages()})()
    schema = {"type": "object", "properties": {}}
    result = AnthropicProvider(client=client).generate(_req(model="claude-x", temperature=0.0))  # no schema
    assert "tools" not in captured  # not forced without a schema

    captured.clear()
    result = AnthropicProvider(client=client).generate(
        GenerationRequest(prompt="p", model="claude-x", condition=Condition.NORMAL, temperature=0.0, json_schema=schema)
    )
    assert captured["tool_choice"]["name"] == "submit_scores"
    assert captured["tools"][0]["input_schema"] is schema
    assert _json.loads(result.output_text) == {"distinctness": {"score": 4}}


def test_anthropic_thinking_judge_uses_adaptive_effort_and_relaxes_tool_choice():
    # A thinking judge (json_schema + THINKING condition) must use the modern adaptive
    # thinking API (Opus 4.8+): thinking.type.adaptive + output_config.effort. It also
    # cannot force a named tool under thinking — only tool_choice "auto" is allowed.
    from calvino.providers import AnthropicProvider
    import json as _json

    captured = {}
    tool_block = type("B", (), {"type": "tool_use", "input": {"distinctness_score": 4}})()
    resp = type("R", (), {"content": [tool_block], "model": "claude-x", "stop_reason": "tool_use", "usage": None})()

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return resp

    client = type("Cl", (), {"messages": _Messages()})()
    schema = {"type": "object", "properties": {}}
    result = AnthropicProvider(client=client).generate(
        GenerationRequest(prompt="p", model="claude-x", condition=Condition.THINKING,
                          temperature=0.0, json_schema=schema, effort="high", max_tokens=1024)
    )
    assert captured["tool_choice"] == {"type": "auto"}            # not forced under thinking
    assert captured["thinking"] == {"type": "adaptive"}           # modern API, not enabled+budget
    assert captured["output_config"] == {"effort": "high"}        # effort knob
    assert "temperature" not in captured                           # adaptive omits temperature (API default 1.0)
    assert captured["max_tokens"] >= 8192                          # headroom for thinking + output
    assert _json.loads(result.output_text) == {"distinctness_score": 4}


def test_anthropic_thinking_falls_back_to_legacy_enabled_when_adaptive_rejected():
    # Older models reject adaptive thinking; the provider must fall back to the legacy
    # enabled + budget_tokens shape and succeed.
    from calvino.providers import AnthropicProvider
    import json as _json

    captured = {}
    tool_block = type("B", (), {"type": "tool_use", "input": {"distinctness_score": 4}})()
    ok = type("R", (), {"content": [tool_block], "model": "old", "stop_reason": "tool_use", "usage": None})()

    class _Messages:
        def create(self, **kwargs):
            if kwargs.get("thinking", {}).get("type") == "adaptive":
                raise RuntimeError("thinking.type.adaptive is not supported for this model")
            captured.update(kwargs)
            return ok

    client = type("Cl", (), {"messages": _Messages()})()
    result = AnthropicProvider(client=client).generate(
        GenerationRequest(prompt="p", model="old", condition=Condition.THINKING,
                          temperature=0.0, json_schema={"type": "object"}, thinking_budget=4096)
    )
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 4096}  # legacy shape
    assert "output_config" not in captured                                     # dropped on fallback
    assert captured["temperature"] == 1.0                                      # legacy path forces 1.0
    assert _json.loads(result.output_text) == {"distinctness_score": 4}


def test_anthropic_thinking_judge_falls_back_to_text_when_no_tool_call():
    # Under tool_choice auto the model may answer in plain text instead of calling the
    # tool; the provider must return that text so the caller's JSON parser can recover it.
    from calvino.providers import AnthropicProvider

    text_block = type("B", (), {"type": "text", "text": '{"distinctness_score": 3}'})()
    resp = type("R", (), {"content": [text_block], "model": "claude-x", "stop_reason": "end_turn", "usage": None})()

    class _Messages:
        def create(self, **kwargs):
            return resp

    client = type("Cl", (), {"messages": _Messages()})()
    result = AnthropicProvider(client=client).generate(
        GenerationRequest(prompt="p", model="claude-x", condition=Condition.THINKING,
                          temperature=0.0, json_schema={"type": "object"}, thinking_budget=4096)
    )
    assert result.output_text == '{"distinctness_score": 3}'


def test_openai_json_schema_sets_response_format():
    client = _FakeOpenAIClient('{"x": 1}')
    schema = {"type": "object"}
    OpenAIProvider(client=client).generate(
        GenerationRequest(prompt="p", model="gpt-4.1", condition=Condition.NORMAL, temperature=0.0, json_schema=schema)
    )
    assert client.last_kwargs["response_format"] == {"type": "json_object"}


def test_google_json_schema_sets_mime():
    client = _FakeGenAIClient('{"x": 1}')
    schema = {"type": "object"}
    GoogleProvider(client=client).generate(
        GenerationRequest(prompt="p", model="gemini-x", condition=Condition.NORMAL, temperature=0.0, json_schema=schema)
    )
    assert client.calls[0]["config"]["response_mime_type"] == "application/json"


def test_get_provider_resolution():
    assert isinstance(get_provider("mock"), MockProvider)
    assert isinstance(get_provider("ollama"), OllamaProvider)
    # 'openai'/'anthropic'/'google' construct real SDK clients that need keys, so
    # they're not instantiated here; the unknown-name path is the contract we check.
    with pytest.raises(ValueError):
        get_provider("nope")
