"""Tests for the generation harness, using the deterministic MockProvider so no
network or API key is needed. Focus: persistence, idempotent cache-skip across
both a single run and a fresh run from disk, and spec fan-out."""

import pytest

from calvino.harness import GenerationHarness, GenerationSpec, expand_specs
from calvino.items import Item
from calvino.models import Condition, Generation, PromptVariant
from calvino.providers import MockProvider
from calvino.storage import GENERATIONS, Store


@pytest.fixture
def item():
    return Item(
        item_id="lantern-01",
        object="The Lantern",
        object_type="tarot_image_object",
        description="a battered tin lantern",
        story_a="A smuggler waits on the headland.",
        story_b="A widow descends into the flooded cellar.",
        prompt_variants=[PromptVariant.BASE, PromptVariant.MINIMAL],
    )


def test_run_spec_persists_generation(tmp_path, item):
    store = Store(tmp_path)
    provider = MockProvider()
    harness = GenerationHarness(store, provider)

    spec = GenerationSpec(item=item, model="claude-x", sample_idx=0)
    gen = harness.run_spec(spec)

    assert gen is not None
    assert gen.id == spec.request_id()
    assert gen.rendered_prompt  # prompt stored verbatim
    assert gen.provider_meta["provider"] == "mock"
    # Persisted and reloadable.
    loaded = store.load(GENERATIONS, Generation)
    assert len(loaded) == 1
    assert loaded[0].id == gen.id


def test_cache_skip_within_run(tmp_path, item):
    store = Store(tmp_path)
    provider = MockProvider()
    harness = GenerationHarness(store, provider)
    spec = GenerationSpec(item=item, model="claude-x", sample_idx=0)

    first = harness.run_spec(spec)
    second = harness.run_spec(spec)  # identical request

    assert first is not None
    assert second is None  # skipped
    assert len(provider.calls) == 1  # provider called only once
    assert len(store.load(GENERATIONS, Generation)) == 1


def test_cache_skip_across_runs_from_disk(tmp_path, item):
    spec = GenerationSpec(item=item, model="claude-x", sample_idx=0)

    # First run writes to disk.
    p1 = MockProvider()
    GenerationHarness(Store(tmp_path), p1).run_spec(spec)
    assert len(p1.calls) == 1

    # A brand-new harness over the same dir rebuilds its cache from disk and skips.
    p2 = MockProvider()
    result = GenerationHarness(Store(tmp_path), p2).run_spec(spec)
    assert result is None
    assert len(p2.calls) == 0  # no API call on resume


def test_different_sample_idx_is_a_distinct_request(tmp_path, item):
    store = Store(tmp_path)
    provider = MockProvider()
    harness = GenerationHarness(store, provider)

    g0 = harness.run_spec(GenerationSpec(item=item, model="claude-x", sample_idx=0))
    g1 = harness.run_spec(GenerationSpec(item=item, model="claude-x", sample_idx=1))

    assert g0 is not None and g1 is not None
    assert g0.id != g1.id
    assert len(store.load(GENERATIONS, Generation)) == 2


def test_thinking_condition_changes_request_id(item):
    normal = GenerationSpec(item=item, model="claude-x", condition=Condition.NORMAL)
    thinking = GenerationSpec(item=item, model="claude-x", condition=Condition.THINKING)
    assert normal.request_id() != thinking.request_id()


def test_run_all_returns_only_new_rows(tmp_path, item):
    store = Store(tmp_path)
    provider = MockProvider()
    harness = GenerationHarness(store, provider)
    specs = [
        GenerationSpec(item=item, model="claude-x", sample_idx=0),
        GenerationSpec(item=item, model="claude-x", sample_idx=1),
        GenerationSpec(item=item, model="claude-x", sample_idx=0),  # dup of first
    ]
    produced = harness.run_all(specs)
    assert len(produced) == 2
    assert len(provider.calls) == 2


def test_expand_specs_cartesian_product(item):
    specs = expand_specs(item, models=["a", "b"], samples=3)
    # 2 variants x 2 models x 1 condition x 3 samples = 12
    assert len(specs) == 12
    assert len({s.request_id() for s in specs}) == 12  # all distinct


def test_api_model_override_calls_endpoint_but_keys_cache_on_name(tmp_path, item):
    """Reasoning-control slice: `gpt-5.5-noreason` is the on-disk identity (cache key +
    stored model) while the real `gpt-5.5` endpoint is called. The override must NOT
    enter the request id, and the provider must receive the api_model."""
    class RecordingProvider:
        def __init__(self):
            self.called_with = []
            self._mock = MockProvider()

        def generate(self, request):
            self.called_with.append(request.model)
            return self._mock.generate(request)

    store = Store(tmp_path)
    provider = RecordingProvider()
    harness = GenerationHarness(store, provider)

    spec = GenerationSpec(item=item, model="gpt-5.5-noreason", api_model="gpt-5.5")
    plain = GenerationSpec(item=item, model="gpt-5.5-noreason")  # no override

    gen = harness.run_spec(spec)
    assert provider.called_with == ["gpt-5.5"]          # endpoint = api_model
    assert gen.model == "gpt-5.5-noreason"              # stored under on-disk name
    assert gen.id == spec.request_id()
    # api_model does not change the cache key: same id with or without the override.
    assert spec.request_id() == plain.request_id()


def test_expand_specs_threads_api_model(item):
    specs = expand_specs(item, models=["gpt-5.5-noreason"], samples=1, api_model="gpt-5.5")
    assert all(s.call_model == "gpt-5.5" for s in specs)
    assert all(s.model == "gpt-5.5-noreason" for s in specs)
    # default (no override) calls the on-disk name
    plain = expand_specs(item, models=["m"], samples=1)
    assert all(s.call_model == "m" for s in plain)
