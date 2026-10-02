"""Generation harness.

Drives one provider over a set of (item, variant, model, condition, sample)
requests, persisting each output as a Generation. The harness is provider-
agnostic: it depends only on the `Provider` protocol.

Caching is idempotent and disk-backed: a generation's id IS the request hash
(`make_request_id`), and the in-memory cache is the set of ids already on disk.
A re-run therefore skips any request already completed — no API call — so the
whole pilot can be resumed cheaply after an interruption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .items import Item
from .models import Condition, Generation, PromptVariant
from .prompts import render_prompt
from .providers import GenerationRequest, Provider
from .storage import GENERATIONS, Store, make_request_id


@dataclass(frozen=True)
class GenerationSpec:
    """A single unit of work: what to generate, once. `label_visible` drives the
    label-ablation rendering and is part of the recorded prompt, not a separate
    cache dimension — distinct prompts already yield distinct outputs."""

    item: Item
    model: str
    variant: PromptVariant = PromptVariant.BASE
    condition: Condition = Condition.NORMAL
    sample_idx: int = 0
    temperature: float = 0.7
    seed: int | None = None
    label_visible: bool = False
    # API model id to actually call, when it differs from the on-disk identity `model`
    # (e.g. `gpt-5.5-noreason` is stored under that name for cache/provenance but calls the
    # real `gpt-5.5` endpoint — the 2026-07-08 reasoning-control). None => call `model`.
    # `model` remains the sole cache dimension; api_model never enters the request id.
    api_model: str | None = None
    # Adaptive-thinking effort for the THINKING condition (Fable/Mythos 5+). Call-time only,
    # like api_model — NOT part of request_id, so effort variants MUST use distinct `model`
    # names or they collide in the cache. Ignored unless condition == THINKING.
    effort: str = "high"

    @property
    def call_model(self) -> str:
        return self.api_model or self.model

    def request_id(self) -> str:
        return make_request_id(
            item_id=self.item.item_id,
            model=self.model,
            condition=self.condition.value,
            prompt_variant=self.variant.value,
            sample_idx=self.sample_idx,
            temperature=self.temperature,
            seed=self.seed,
        )


class GenerationHarness:
    def __init__(self, store: Store, provider: Provider):
        self.store = store
        self.provider = provider
        self._done: set[str] = store.generation_ids()

    def run_spec(self, spec: GenerationSpec) -> Generation | None:
        """Generate one output, or return None if it is already on disk.

        On a cache hit nothing is called or written. On a miss the provider is
        invoked, the Generation is appended to disk, and its id is recorded so a
        later spec with the same request is skipped within this run too."""
        request_id = spec.request_id()
        if request_id in self._done:
            return None

        prompt = render_prompt(spec.item, spec.variant, label_visible=spec.label_visible)
        result = self.provider.generate(
            GenerationRequest(
                prompt=prompt,
                model=spec.call_model,  # api_model override if set, else the on-disk name
                condition=spec.condition,
                temperature=spec.temperature,
                seed=spec.seed,
                effort=spec.effort,  # only consumed when condition == THINKING
            )
        )

        generation = Generation(
            id=request_id,
            item_id=spec.item.item_id,
            object=spec.item.object,
            object_type=spec.item.object_type,
            model=spec.model,
            condition=spec.condition,
            prompt_variant=spec.variant,
            sample_idx=spec.sample_idx,
            temperature=spec.temperature,
            seed=spec.seed,
            rendered_prompt=prompt,
            output_text=result.output_text,
            provider_meta=result.provider_meta,
        )
        self.store.append(GENERATIONS, generation)
        self._done.add(request_id)
        return generation

    def run_all(
        self,
        specs: list[GenerationSpec],
        progress: Optional[Callable[[GenerationSpec, Optional[Generation]], None]] = None,
    ) -> list[Generation]:
        """Run a batch, skipping cached specs. Returns only newly generated rows.
        `progress(spec, generation_or_None)` is called after each spec (generation
        is None on a cache hit) for progress reporting."""
        produced: list[Generation] = []
        for spec in specs:
            generation = self.run_spec(spec)
            if generation is not None:
                produced.append(generation)
            if progress is not None:
                progress(spec, generation)
        return produced


def expand_specs(
    item: Item,
    *,
    models: list[str],
    samples: int,
    conditions: list[Condition] | None = None,
    temperature: float = 0.7,
    api_model: str | None = None,
    effort: str = "high",
) -> list[GenerationSpec]:
    """Fan one item out across its declared prompt variants, the given models,
    conditions, and sample count — the cartesian product that makes up a cell of
    the design. Kept deliberately simple; ablations are added by the caller.

    `api_model` overrides the endpoint actually called for every spec here while the
    on-disk `model` name (the cache key) is preserved — used by the reasoning-control
    slice (`gpt-5.5-noreason` stored, `gpt-5.5` called). Typically the caller passes one
    model per call, so a single override applies cleanly."""
    conditions = conditions or [Condition.NORMAL]
    specs: list[GenerationSpec] = []
    for variant in item.prompt_variants:
        for model in models:
            for condition in conditions:
                for sample_idx in range(samples):
                    specs.append(
                        GenerationSpec(
                            item=item,
                            model=model,
                            variant=variant,
                            condition=condition,
                            sample_idx=sample_idx,
                            temperature=temperature,
                            api_model=api_model,
                            effort=effort,
                        )
                    )
    return specs
