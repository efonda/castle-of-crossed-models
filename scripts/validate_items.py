#!/usr/bin/env python3
"""Item audit by LLM reviewers from other labs (paper §8). CALLS MODEL APIS.

Each item is reviewed read-only against a fixed rubric (gates A/B/C, dimensions D1-D7) by
gemini-3.1-pro (Google) and grok-4.3 (xAI), labs other than the drafting assistant's. The
aggregation rule is conservative: one reviewer's leniency cannot clear an item the other flags.
This script never writes to data/items*.yaml; flags are recorded, not fixed. Results:
data/runs/item-validation/ and data/item_validation_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calvino.models import Condition  # noqa: E402
from calvino.providers import GenerationRequest, ProviderError, get_provider  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUBRIC = ROOT / "reports" / "item-validation-rubric.md"
OUT_DIR = ROOT / "data" / "runs" / "item-validation"
RAW = OUT_DIR / "raw_validation_responses.jsonl"
LOG_CSV = ROOT / "data" / "item_validation_log.csv"
SUMMARY = OUT_DIR / "summary.json"

BANKS = {
    "signal": "items.yaml",
    "hard": "items_hard.yaml",
    "italian": "items_it.yaml",
}

# reviewer key -> (provider name, model id, lineage)
REVIEWERS = {
    "gemini": ("google", "gemini-3.1-pro-preview", "Google"),
    "grok": ("xai", "grok-4.3", "xAI"),
}

DIMS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]


class Review(BaseModel):
    """One reviewer's verdict on one item. Validated at the parse boundary."""

    item_id: str
    bank: str
    gateA: Literal["pass", "fail"]
    gateB: Literal["pass", "fail"]
    gateC: Literal["pass", "fail"]
    gateB_bridge_sketch: str = ""
    gateC_surface_forms: list[str] = Field(default_factory=list)
    D1: int
    D2: int
    D3: int
    D4: int | None = None
    D5: int
    D6: int | None = None
    D7: int
    rationale: str = ""
    worst_case: str = ""

    def verdict(self) -> str:
        """The rubric's pre-committed per-review decision rule."""
        if "fail" in (self.gateA, self.gateB, self.gateC):
            return "flagged"
        scores = [getattr(self, d) for d in DIMS if getattr(self, d) is not None]
        if any(s == 0 for s in scores):
            return "flagged"
        if sum(1 for s in scores if s == 1) >= 3:
            return "weak"
        return "validated"


PROMPT = """You are auditing one item from a creative-writing benchmark. The item gives a physical \
OBJECT and two story SEEDS; a model under test must write exactly four sentences in which sentences \
1-2 tell story A, sentences 3-4 tell story B, both stories feature the object, the object MEANS \
SOMETHING GENUINELY DIFFERENT in each, and the two halves are joined by a grounded physical, \
causal, or temporal link. Coincidence, dream, telepathy, and pure symbolism are disallowed as the \
link.

Your job is to judge the ITEM, not any model's output. You will never see model outputs. A sound \
item is one where a capable writer could satisfy the task and a weak writer would visibly fail; an \
unsound item makes low scores measure the item instead of the model.

ITEM
  item_id: {item_id}
  bank: {bank}
  language: {language}
  object: {object}
  description: {description}
  story A seed: {story_a}
  story B seed: {story_b}

GATES (pass/fail)
A Contamination. Object, seeds, and phrasing are original: not lifted from Calvino's "The Castle of
  Crossed Destinies", not a recognizable episode from Calvino or another well-known work, not a
  famous named artifact (Excalibur-class objects fail).
B Structural feasibility. Each seed can be told in two sentences, and you can sketch at least one
  grounded physical/causal/temporal bridge between the halves in under a minute. State the bridge
  you sketched. If no grounded bridge exists, the item forces the disallowed moves.
C Gate compatibility. The deterministic validator matches the object by surface form. Confirm the
  object's natural surface forms are matchable (e.g. "The Salt Compass" will appear as "compass").
  A multiword object that models paraphrase produces false gate failures that look like model
  errors.

DIMENSIONS (0 = fail, 1 = usable with reservation, 2 = sound)
D1 Seed independence. Two genuinely distinct narratives (different agents, stakes, situations), not
   two beats of one plot. 2 fully independent; 1 related but distinguishable; 0 one story split.
D2 Object bivalence. The object admits at least two clearly different meanings without changing
   what it physically is. 2 multiple natural construals; 1 second meaning strained; 0 monovalent.
D3 Difficulty placement. Neither trivially solvable by a mid-tier model nor impossible in four
   sentences. 2 challenging but fair; 1 borderline; 0 FRAGILE: likely to trip many models on a
   technicality and dominate the bank's variance. SCORE THIS HARSHLY. One fragile item moved a
   cluster reliability figure from 0.142 to 0.574; fragility is insidious because judges agreeing
   on a technicality looks like signal.
{d4_block}
D5 Neutrality of register. Seeds are flat informational scaffolding, not a style that rewards one
   model family's known register. 2 neutral; 1 mild lean; 0 the seed is half the passage.
{d6_block}
D7 Bridge non-dictation. The seeds must not PRE-WRITE the bridge. If one bridge is dictated by the
   seeds, the crossing reduces to completion and originality has no headroom. 2 several distinct
   bridges plausible; 1 one obvious bridge plus strained alternatives; 0 the seeds dictate it.

This audit is READ-ONLY. Do not propose rewrites, do not suggest replacement objects or seeds.
Judge the item as given. If it is unsound, say so and say which gate or dimension fails.

Return ONLY this JSON, no prose before or after:
{{
  "item_id": "{item_id}", "bank": "{bank}",
  "gateA": "pass" or "fail", "gateB": "pass" or "fail", "gateC": "pass" or "fail",
  "gateB_bridge_sketch": "the grounded bridge you constructed, one sentence",
  "gateC_surface_forms": ["the natural surface forms you expect"],
  "D1": 0, "D2": 0, "D3": 0, "D4": {d4_default}, "D5": 0, "D6": {d6_default}, "D7": 0,
  "rationale": "one line, naming the weakest element",
  "worst_case": "the most likely way this item produces a low score for a reason that is not the \
model's fault, or 'none'"
}}"""


D4_BLOCK = """D4 Trap/invention integrity. Cliche-traps: the stock symbolism is strong enough that the
   default reading actually falls in - a trap nobody would spring is not a trap. Invented objects:
   genuinely no cultural prior; the name does not collide with an existing symbol, brand, or
   mythological item in English or Italian. 2 mechanism bites; 1 partial; 0 inert."""
D4_SKIP = "D4 Trap/invention integrity. NOT APPLICABLE to this bank. Return null."
D6_BLOCK = """D6 Language quality. Idiomatic, natural, unambiguous Italian; no calques; the object
   keeps its intended bivalence in Italian."""
D6_SKIP = "D6 Language quality. NOT APPLICABLE to this bank (English item). Return null."


def render(item: dict, bank: str) -> str:
    d4_applies = bank == "hard"
    d6_applies = bank == "italian"
    return PROMPT.format(
        item_id=item["item_id"],
        bank=bank,
        language=item.get("language", "en"),
        object=item["object"],
        description=" ".join(item["description"].split()),
        story_a=" ".join(item["story_a"].split()),
        story_b=" ".join(item["story_b"].split()),
        d4_block=D4_BLOCK if d4_applies else D4_SKIP,
        d6_block=D6_BLOCK if d6_applies else D6_SKIP,
        d4_default="0" if d4_applies else "null",
        d6_default="0" if d6_applies else "null",
    )


def extract_json(text: str) -> dict:
    """First balanced {...} block. Tolerates fenced code and surrounding prose."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object")


def aggregate(reviews: list[Review]) -> tuple[str, str]:
    """PRE-COMMITTED aggregation rule. Conservative: any reviewer's flag flags the item.

    flagged  if ANY reviewer fails a gate or scores any dimension 0
    weak     else if ANY reviewer returns >= 3 dimensions at 1
    validated otherwise
    Disagreements are reported as counts, never averaged into a middle verdict.
    """
    verdicts = [r.verdict() for r in reviews]
    if "flagged" in verdicts:
        agg = "flagged"
    elif "weak" in verdicts:
        agg = "weak"
    else:
        agg = "validated"
    note = "unanimous" if len(set(verdicts)) == 1 else "split: " + ", ".join(verdicts)
    return agg, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="*", default=list(BANKS), choices=list(BANKS))
    ap.add_argument("--reviewers", nargs="*", default=list(REVIEWERS), choices=list(REVIEWERS))
    ap.add_argument("--dry-run", action="store_true", help="render prompts, make no API calls")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rubric_hash = hashlib.sha256(RUBRIC.read_bytes()).hexdigest()

    # resume: (item_id, reviewer) pairs already on disk
    done: set[tuple[str, str]] = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["item_id"], rec["reviewer"]))
    if done:
        print(f"resume: {len(done)} (item, reviewer) pairs already on disk\n")

    work = []
    for bank in args.banks:
        data = yaml.safe_load((ROOT / "data" / BANKS[bank]).read_text())
        for item in data.get("items", []):
            for rev in args.reviewers:
                work.append((bank, item, rev))

    if args.dry_run:
        bank, item, rev = work[0]
        print(f"--- example prompt: {item['item_id']} / {rev} ---\n")
        print(render(item, bank))
        print(f"\n{len(work)} (item, reviewer) calls would be made; "
              f"{len([w for w in work if (w[1]['item_id'], w[2]) not in done])} not yet on disk.")
        return 0

    providers: dict[str, object] = {}
    parsed: dict[str, list[Review]] = {}
    parse_failures = 0

    for bank, item, rev in work:
        key = (item["item_id"], rev)
        if key in done:
            continue
        prov_name, model, _ = REVIEWERS[rev]
        if rev not in providers:
            providers[rev] = get_provider(prov_name)
        prompt = render(item, bank)
        try:
            result = providers[rev].generate(GenerationRequest(
                prompt=prompt, model=model, condition=Condition.NORMAL,
                temperature=0.0, max_tokens=args.max_tokens,
            ))
        except ProviderError as exc:
            print(f"  SKIP  {item['item_id']:<22} {rev:<7} provider error: {exc}")
            continue

        # verbatim BEFORE parsing
        with RAW.open("a") as fh:
            fh.write(json.dumps({
                "item_id": item["item_id"], "bank": bank, "reviewer": rev, "model": model,
                "rubric_sha256": rubric_hash, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_response": result.output_text,
                "provider_meta": result.provider_meta,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n")

        try:
            review = Review(**{**extract_json(result.output_text),
                               "item_id": item["item_id"], "bank": bank})
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            parse_failures += 1
            print(f"  PARSE FAIL  {item['item_id']:<22} {rev:<7} {type(exc).__name__}")
            continue
        parsed.setdefault(item["item_id"], []).append(review)
        print(f"  {item['item_id']:<22} {rev:<7} {review.verdict()}")

    # rebuild the full parsed set from disk so a resumed run writes a complete log
    parsed = {}
    for line in RAW.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            r = Review(**{**extract_json(rec["raw_response"]),
                          "item_id": rec["item_id"], "bank": rec["bank"]})
        except Exception:
            continue
        parsed.setdefault(rec["item_id"], []).append((rec["reviewer"], r))

    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for item_id, pairs in parsed.items():
        reviews = [r for _, r in pairs]
        agg, note = aggregate(reviews)
        for reviewer, r in pairs:
            rows.append({
                "item_id": item_id, "bank": r.bank, "reviewer": reviewer,
                "gateA": r.gateA, "gateB": r.gateB, "gateC": r.gateC,
                **{d: ("" if getattr(r, d) is None else getattr(r, d)) for d in DIMS},
                "verdict": r.verdict(), "aggregate_verdict": agg, "agreement": note,
                "rationale": r.rationale.replace("\n", " ")[:300],
                "worst_case": r.worst_case.replace("\n", " ")[:300],
                "action_taken": "none (read-only audit)" if agg != "flagged"
                                else "flag -> reported sensitivity cut",
                "date": today,
            })

    if rows:
        fields = list(rows[0])
        with LOG_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["bank"], r["item_id"], r["reviewer"])))

    per_item = {i: aggregate([r for _, r in p])[0] for i, p in parsed.items()}
    counts: dict[str, int] = {}
    for v in per_item.values():
        counts[v] = counts.get(v, 0) + 1
    summary = {
        "reviewers": {k: {"provider": v[0], "model": v[1], "lineage": v[2],
                          "role": "out-of-lineage LLM reviewer, read-only"}
                      for k, v in REVIEWERS.items() if k in args.reviewers},
        "rubric": {"path": str(RUBRIC.relative_to(ROOT)), "sha256": rubric_hash},
        "banks": args.banks,
        "items_reviewed": len(per_item),
        "aggregate_counts": counts,
        "flagged_items": sorted(i for i, v in per_item.items() if v == "flagged"),
        "weak_items": sorted(i for i, v in per_item.items() if v == "weak"),
        "splits": sorted(i for i, p in parsed.items()
                         if len({r.verdict() for _, r in p}) > 1),
        "parse_failures_this_run": parse_failures,
        "aggregation_rule": aggregate.__doc__.strip().splitlines()[0],
        "read_only": "no item was added, edited, or dropped by this pass",
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{len(per_item)} items, {len(rows)} review rows")
    for v, n in sorted(counts.items()):
        print(f"  {v:<10} {n}")
    if summary["flagged_items"]:
        print(f"  FLAGGED: {', '.join(summary['flagged_items'])}")
        print("  -> each becomes a reported sensitivity cut: recompute headline rates with the")
        print("     item excluded and report both numbers (rubric decision rule).")
    if summary["splits"]:
        print(f"  reviewer disagreement on: {', '.join(summary['splits'])}")
    print(f"\nlog     {LOG_CSV.relative_to(ROOT)}")
    print(f"raw     {RAW.relative_to(ROOT)}")
    print(f"summary {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
