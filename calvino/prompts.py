"""Prompt renderer.

Templates are stored as constants and filled per item. The exact rendered prompt
is persisted with every generation (never reconstructed at analysis time), so the
template wording here is itself part of the experimental record — change it only
with a version bump to the run.

BASE is reproduced verbatim from brief §7. MINIMAL compresses the constraint list
into prose to test prompt sensitivity (run for 2-3 items). HARDER is the optional
ablation requiring the object to *cause* the transition.
"""

from __future__ import annotations

from .items import Item
from .models import PromptVariant

BASE_TEMPLATE = """\
You will write one single paragraph where two separate stories cross.

The same object appears in both stories, but it must mean something different in each.

Crossing object: {OBJECT} — {DESCRIPTION}

Story A, ending here: {STORY_A_CONTEXT}

Story B, beginning here: {STORY_B_CONTEXT}

Constraints:
1. Write exactly 4 sentences.
2. Sentences 1–2 must close Story A.
3. Sentences 3–4 must begin Story B.
4. The object must connect the two stories through one grounded physical, causal, or temporal link.
5. Symbolic resonance is allowed, but it cannot replace story-world causality; the reader must see how the object physically, causally, or temporally belongs to both stories.
6. The object must mean one thing in Story A and a clearly different thing in Story B, without changing what the object physically is.
7. Use concrete physical detail. Avoid abstract moralising, melodrama, and generic fantasy language.

Output only the paragraph."""

MINIMAL_TEMPLATE = """\
Write exactly four sentences forming one paragraph in which two separate stories cross through a single shared object. The first two sentences close Story A; the last two begin Story B. The object — {OBJECT} ({DESCRIPTION}) — must mean one thing in Story A and something clearly different in Story B without changing what it physically is, and the two halves must be joined by a grounded physical, causal, or temporal link rather than coincidence or symbolism alone. Use concrete detail; avoid melodrama and generic fantasy language.

Story A, ending here: {STORY_A_CONTEXT}

Story B, beginning here: {STORY_B_CONTEXT}

Output only the paragraph."""

# HARDER adds one extra causal constraint to the base prompt.
_HARDER_EXTRA = """
8. The same object must physically *cause* the transition between the two stories, not merely appear in both."""

HARDER_TEMPLATE = BASE_TEMPLATE.replace(
    "\n\nOutput only the paragraph.",
    _HARDER_EXTRA + "\n\nOutput only the paragraph.",
)

# --- Free-sentence-count style variants -------------------------------------
# Three variants on the gravita<->leggerezza axis, all with FREE sentence count (the
# fixed-4-sentence rule structurally forces long dense sentences = gravita). NOT part of
# the scored benchmark — different prompt + free count, decoupled by design.

LIGHT_MIN_TEMPLATE = """\
Write one short paragraph in which two separate stories cross through a single shared object that means something different in each, joined by a grounded physical, causal, or temporal link (not coincidence or symbolism alone). The first part closes Story A; the second begins Story B.

Crossing object: {OBJECT} — {DESCRIPTION}
Story A, ending here: {STORY_A_CONTEXT}
Story B, beginning here: {STORY_B_CONTEXT}

Write it light, in the spirit of Calvino.

Output only the paragraph."""

LIGHT_MAX_TEMPLATE = """\
You will write one short paragraph where two separate stories cross.

The same object appears in both stories, but it must mean something different in each.

Write in the spirit of Italo Calvino's Six Memos for the Next Millennium — his values for literature: lightness (leggerezza), exactitude (esattezza), quickness (rapidità), visibility (visibilità), and multiplicity (molteplicità). Aim for those qualities, not for an imitation of Calvino's voice or settings.

Crossing object: {OBJECT} — {DESCRIPTION}

Story A, ending here: {STORY_A_CONTEXT}

Story B, beginning here: {STORY_B_CONTEXT}

Form:
- Write a short paragraph — as many sentences as the crossing needs and no more. Short sentences are welcome; let the prose move.
- The first part closes Story A; the second part begins Story B.
- The object must connect the two stories through one grounded physical, causal, or temporal link, and that link should feel quick and inevitable, not laboured.
- The object must mean one thing in Story A and a clearly different thing in Story B, without changing what it physically is. Symbolic resonance is allowed but cannot replace story-world causality; the reader must see how the object belongs to both stories.

Aim, in Calvino's terms:
- Lightness — let the mind leap between the object's two meanings; prefer a few precisely chosen images to piled detail; be crystalline, not encrusted; favour lucidity and wit over elegiac weight. (Image density and sentence length are separate: choose few, exact images, but length is neutral — a long sentence can be light. Lightness is agility of thought, not a low word-count.)
- Exactitude — the precise word and the well-defined image; nothing vague or approximate.
- Quickness — economy and momentum; no digression; the crossing should move.
- Visibility — a sharp image the reader sees in the mind's eye.
- Multiplicity — let the one object be the node where the two stories' meanings meet and turn; this crossing is itself an exercise in Calvino's multiplicity.

Output only the paragraph."""

GRAVITA_FREE_TEMPLATE = """\
You will write one short paragraph where two separate stories cross.

The same object appears in both stories, but it must mean something different in each.

Crossing object: {OBJECT} — {DESCRIPTION}

Story A, ending here: {STORY_A_CONTEXT}

Story B, beginning here: {STORY_B_CONTEXT}

Constraints:
1. Write a short paragraph — as many sentences as the crossing needs and no more.
2. The first part closes Story A; the second part begins Story B.
3. The object must connect the two stories through one grounded physical, causal, or temporal link.
4. Symbolic resonance is allowed, but it cannot replace story-world causality; the reader must see how the object physically, causally, or temporally belongs to both stories.
5. The object must mean one thing in Story A and a clearly different thing in Story B, without changing what the object physically is.
6. Use concrete physical detail. Avoid abstract moralising, melodrama, and generic fantasy language.

Output only the paragraph."""


# --- Italian arm (generalization check) -------------------------------------
# Faithful translation of BASE_TEMPLATE. The instruction must be in the target
# language so the model writes Italian. NOTE: this prompt is part of the
# experimental record — have it NATIVE-reviewed before a live Italian run, same
# as the items. Only BASE is provided for Italian (the arm runs BASE only).
BASE_TEMPLATE_IT = """\
Scriverai un unico paragrafo in cui due storie separate si incrociano.

Lo stesso oggetto compare in entrambe le storie, ma deve significare qualcosa di diverso in ciascuna.

Oggetto del crocevia: {OBJECT} — {DESCRIPTION}

Storia A, che finisce qui: {STORY_A_CONTEXT}

Storia B, che comincia qui: {STORY_B_CONTEXT}

Vincoli:
1. Scrivi esattamente 4 frasi.
2. Le frasi 1–2 devono chiudere la Storia A.
3. Le frasi 3–4 devono iniziare la Storia B.
4. L'oggetto deve collegare le due storie attraverso un unico legame concreto, fisico, causale o temporale.
5. La risonanza simbolica è ammessa, ma non può sostituire la causalità del mondo narrativo; il lettore deve poter vedere come l'oggetto appartiene fisicamente, causalmente o temporalmente a entrambe le storie.
6. L'oggetto deve significare una cosa nella Storia A e una cosa chiaramente diversa nella Storia B, senza cambiare ciò che l'oggetto fisicamente è.
7. Usa dettagli fisici concreti. Evita il moralismo astratto, il melodramma e il linguaggio fantasy generico.

Scrivi solo il paragrafo."""

# Language -> {variant -> template}. English carries all variants; Italian runs BASE only.
_TEMPLATES: dict[str, dict[PromptVariant, str]] = {
    "en": {
        PromptVariant.BASE: BASE_TEMPLATE,
        PromptVariant.MINIMAL: MINIMAL_TEMPLATE,
        PromptVariant.HARDER: HARDER_TEMPLATE,
        PromptVariant.LIGHT_MIN: LIGHT_MIN_TEMPLATE,
        PromptVariant.LIGHT_MAX: LIGHT_MAX_TEMPLATE,
        PromptVariant.GRAVITA_FREE: GRAVITA_FREE_TEMPLATE,
    },
    "it": {
        PromptVariant.BASE: BASE_TEMPLATE_IT,
    },
}


def render_prompt(item: Item, variant: PromptVariant, *, label_visible: bool = False) -> str:
    """Fill a template for an item under a given variant, in the item's language.

    When `label_visible` is True the object is presented by its divinatory label
    (the label ablation); this requires the item to define `label`."""
    if label_visible:
        if item.label is None:
            raise ValueError(f"item {item.item_id!r} has no label for label-visible rendering")
        object_str = item.label
    else:
        object_str = item.object

    lang_templates = _TEMPLATES.get(item.language)
    if lang_templates is None:
        raise ValueError(f"no prompt templates for language {item.language!r}")
    template = lang_templates.get(variant)
    if template is None:
        raise ValueError(f"language {item.language!r} has no {variant.value!r} template")
    return template.format(
        OBJECT=object_str,
        DESCRIPTION=item.description,
        STORY_A_CONTEXT=item.story_a,
        STORY_B_CONTEXT=item.story_b,
    )
