"""Judge-free originality metric (paper §6). CALLS AN EMBEDDING MODEL.

Extracts each generation's construal of its object and scores originality as 1 - cosine
similarity to the nearest stock reading, on the existing generations. The embedder is
injectable so the same logic runs under different embedding families; extractor and embedder
agreement are reported. Results in data/runs/route_a/ (the embeddings cache is not shipped and is
rebuilt on a re-run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calvino.originality import (
    REFERENCE_SETS,
    DistanceRecord,
    distance_rows,
    frontier_separation,
    generation_level_correlation,
    group_contrast,
    model_level_spearman,
    nearest_cliche_distance,
    normalize_within_item,
    per_model_mean,
    resolution_ladder,
    two_embedder_agreement,
    validate_against_tiers,
)
from calvino.construal import (
    CONSTRUAL_SCHEMA,
    construal_distance,
    parse_construal,
    render_construal_prompt,
)

# Default analysis targets (match the report). Worst model = the coherence-confound
# canary; trio = the flagship cluster; control = the known flagship>flash gap.
WORST = "llama3.2:3b"
TRIO = ["gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"]
CONTROL = "gemini-3.5-flash"
JUDGES = {"claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"}

# Resolution ladder (§ signal diagnostics): trio vs opponents at increasing originality-gap.
# Tests whether the metric resolves LARGE gaps even if it misses the moderate flash gap —
# i.e. directly checks "is the flash control just too close?"
LADDER = [
    ("flash (moderate gap)", ["gemini-3.5-flash"]),
    ("lite (larger gap)", ["claude-haiku-4-5-20251001", "gemini-3.1-flash-lite"]),
    ("local/worst (largest gap)", ["gemma4:e4b", "gpt-oss:20b", "llama3.2:3b"]),
]


def _log(m): print(m, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Embedder backends (lazy imports; keys read from env only — never config)
# ---------------------------------------------------------------------------
class STEmbedder:
    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy optional dep
        self.name = f"st:{model}"; self._m = SentenceTransformer(model)
    def embed(self, texts): return np.asarray(self._m.encode(texts, normalize_embeddings=False), dtype=float)


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY not set (export it in your terminal)")
        self.name = f"openai:{model}"; self._c = OpenAI(); self._model = model
    def embed(self, texts):
        vecs = []
        for batch in _chunked(list(texts), 256):
            out = self._c.embeddings.create(model=self._model, input=batch)
            vecs.extend(d.embedding for d in out.data)
        return np.asarray(vecs, dtype=float)


class GeminiEmbedder:
    def __init__(self, model="gemini-embedding-2", task_type="SEMANTIC_SIMILARITY"):
        from google import genai
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key: raise SystemExit("GOOGLE_API_KEY / GEMINI_API_KEY not set (export it in your terminal)")
        self.name = f"gemini:{model}"; self._c = genai.Client(api_key=key)
        self._model = model; self._task = task_type
    def embed(self, texts):
        from google.genai import types
        cfg = types.EmbedContentConfig(task_type=self._task) if self._task else None

        def _call(contents):
            try:
                return self._c.models.embed_content(model=self._model, contents=contents, config=cfg)
            except TypeError:                       # older SDK: no config kwarg
                return self._c.models.embed_content(model=self._model, contents=contents)

        texts = list(texts)
        vecs = []
        for batch in _chunked(texts, 100):          # Gemini caps batches at 100 requests
            emb = list(_call(batch).embeddings)
            if len(emb) == len(batch):              # model honoured the batch
                vecs.extend(e.values for e in emb)
            else:                                   # some models/SDKs don't batch — one per call
                vecs.extend(_call(t).embeddings[0].values for t in batch)
        return np.asarray(vecs, dtype=float)


class StubEmbedder:
    """Lexical token-hashing into a fixed dim. NOT semantic — for wiring/dry-run only."""
    def __init__(self, dim=256):
        self.name = "stub:lexical-hash"; self.dim = dim
    def embed(self, texts):
        import re
        v = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z']+", t.lower()):
                v[i, hash(w) % self.dim] += 1.0
        return v


def make_embedder(kind: str, model: str | None = None):
    cls = {"st": STEmbedder, "openai": OpenAIEmbedder, "gemini": GeminiEmbedder, "stub": StubEmbedder}[kind]
    return cls(model=model) if (model and kind != "stub") else cls()


class CachingEmbedder:
    """Wraps any embedder with a content-hash cache (key = embedder-name + text), so a
    re-run only embeds *new* text. Keyed by `inner.name` (which includes the model id), so
    different embedders/models never collide. Persisted to embeddings_cache.jsonl in the
    run dir — makes the slow per-text Gemini re-embeds vanish on re-runs."""

    def __init__(self, inner, cache_path: Path):
        self.inner = inner
        self.name = inner.name
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            for l in self.cache_path.read_text().splitlines():
                if l.strip():
                    d = json.loads(l); self._cache[d["key"]] = d["vec"]

    def _key(self, text: str) -> str:
        import hashlib
        return hashlib.sha256((self.name + "\x00" + text).encode("utf-8")).hexdigest()

    def embed(self, texts):
        texts = list(texts)
        uniq_missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._cache))
        if uniq_missing:
            vecs = self.inner.embed(uniq_missing)
            if len(vecs) != len(uniq_missing):
                raise SystemExit(f"[routeA] inner embedder returned {len(vecs)} vectors for "
                                 f"{len(uniq_missing)} texts — aborting.")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                for t, v in zip(uniq_missing, vecs):
                    vv = [float(x) for x in v]
                    self._cache[self._key(t)] = vv
                    f.write(json.dumps({"key": self._key(t), "vec": vv}) + "\n")
            _log(f"[routeA] embed-cache ({self.name}): {len(uniq_missing)} new, "
                 f"{len(texts) - len(uniq_missing)} reused")
        return np.asarray([self._cache[self._key(t)] for t in texts], dtype=float)


# ---------------------------------------------------------------------------
def load_data(run_dir: Path):
    gens = [json.loads(l) for l in (run_dir / "generations.jsonl").read_text().splitlines() if l.strip()]
    orig_acc: dict[str, list[float]] = {}
    for l in (run_dir / "judge_scores.jsonl").read_text().splitlines():
        if not l.strip(): continue
        d = json.loads(l)
        if d["judge_model"] in JUDGES:
            orig_acc.setdefault(d["generation_id"], []).append(d["scores"]["originality"])
    judged = {k: float(np.mean(v)) for k, v in orig_acc.items()}
    return gens, judged


def _ref_vecs(embedder, reference):
    rv_by_item = {}
    for item, refs in reference.items():
        rv = embedder.embed(refs)
        if len(rv) != len(refs):
            raise SystemExit(f"[routeA] embedder returned {len(rv)} vectors for {len(refs)} "
                             f"cliché refs of {item!r} — embedder is not 1-vector-per-input; aborting.")
        rv_by_item[item] = rv
    return rv_by_item


def distances_for(embedder, gens, judged, reference) -> tuple[list[DistanceRecord], dict[str, float]]:
    """Embed cliché refs (once per item) + all generations, compute nearest-cliché distance."""
    ref_vecs = _ref_vecs(embedder, reference)
    texts = [g["output_text"] for g in gens]
    gv = embedder.embed(texts)
    if len(gv) != len(texts):    # guard: never silently zip-truncate (this caught the gemini-embedding-2 batch bug)
        raise SystemExit(f"[routeA] embedder returned {len(gv)} vectors for {len(texts)} generations "
                         f"— would silently truncate; aborting. Check the embedder's batch behaviour.")
    recs, by_gid = [], {}
    for g, vec in zip(gens, gv):
        item = g["item_id"]
        if item not in ref_vecs:
            continue
        dist = float(nearest_cliche_distance(vec[None, :], ref_vecs[item])[0])
        recs.append(DistanceRecord(model=g["model"], item=item, generation_id=g["id"],
                                   distance=dist, judged_originality=judged.get(g["id"], float("nan"))))
        by_gid[g["id"]] = dist
    return recs, by_gid


# ---------------------------------------------------------------------------
# Construal extraction (§11.2 / calvino/construal.py) — the dilution check
# ---------------------------------------------------------------------------
def make_extractor(kind: str, model: str | None):
    """Return a callable (object, passage) -> (meaning_a, meaning_b), plus a `.name`."""
    if kind == "stub":
        import re
        def _stub(obj, passage):
            sents = re.split(r"(?<=[.!?])\s+", passage.strip())
            a = " ".join(sents[:2])[:80] if sents else passage[:80]
            b = " ".join(sents[2:])[:80] if len(sents) > 2 else passage[-80:]
            return (a or obj, b or obj)
        _stub.name = "stub:passage-split"      # NON-semantic; wiring/dry-run only
        return _stub
    from calvino.providers import get_provider, GenerationRequest
    from calvino.models import Condition
    from calvino.judge import _extract_json
    prov = get_provider(kind)
    name = f"{kind}:{model}" if model else kind
    def _extract(obj, passage):
        res = prov.generate(GenerationRequest(prompt=render_construal_prompt(obj, passage),
                                              model=model, condition=Condition.NORMAL,
                                              temperature=0.0, json_schema=CONSTRUAL_SCHEMA))
        return parse_construal(_extract_json(res.output_text))
    _extract.name = name
    return _extract


def extract_construals(extractor, ext_name, gens, out_dir):
    """Extract (meaning_a, meaning_b) per generation, **cached idempotently** per
    (generation, extractor) in construals.jsonl — extraction is the API/expensive step, so
    a re-run never re-calls what is already on disk."""
    path = Path(out_dir) / "construals.jsonl"
    cache = {}
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                d = json.loads(l); cache[(d["generation_id"], d["extractor"])] = (d["meaning_a"], d["meaning_b"])
    out, new = {}, []
    for i, g in enumerate(gens, 1):
        key = (g["id"], ext_name)
        if key in cache:
            out[g["id"]] = cache[key]; continue
        a, b = extractor(g["object"], g["output_text"])
        out[g["id"]] = (a, b)
        new.append({"generation_id": g["id"], "extractor": ext_name, "model": g["model"],
                    "item": g["item_id"], "meaning_a": a, "meaning_b": b})
        if i % 20 == 0: _log(f"[routeA] extracted {i}/{len(gens)} construals …")
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for r in new: f.write(json.dumps(r) + "\n")
    _log(f"[routeA] construals: {len(out)} total ({len(new)} new, {len(out)-len(new)} cached) → {path}")
    return out


def construal_distances_for(embedder, gens, judged, reference, construals):
    """Like `distances_for`, but distance = min nearest-cliché distance over the two
    *extracted construals* (construal.construal_distance), isolating the meaning-bearing
    signal that whole-passage embedding dilutes."""
    ref_vecs = _ref_vecs(embedder, reference)
    pairs = [construals[g["id"]] for g in gens]
    flat = [m for ab in pairs for m in ab]                 # [a0,b0,a1,b1,…]
    fv = embedder.embed(flat)
    if len(fv) != len(flat):
        raise SystemExit(f"[routeA] embedder returned {len(fv)} vectors for {len(flat)} construals — aborting.")
    recs, by_gid = [], {}
    for i, g in enumerate(gens):
        item = g["item_id"]
        if item not in ref_vecs:
            continue
        dist = construal_distance(fv[2 * i:2 * i + 2], ref_vecs[item])
        recs.append(DistanceRecord(model=g["model"], item=item, generation_id=g["id"],
                                   distance=dist, judged_originality=judged.get(g["id"], float("nan"))))
        by_gid[g["id"]] = dist
    return recs, by_gid


def signal_diagnostics(records) -> dict:
    """STEP 5 bundle — computed once, used for both printing and persistence so the
    on-disk summary carries them (no need to copy numbers out of the terminal)."""
    return {
        "generation_level": generation_level_correlation(records),
        "model_level": model_level_spearman(records),
        "resolution_ladder": resolution_ladder(records, TRIO, LADDER),
        "within_item_normalized": generation_level_correlation(normalize_within_item(records)),
    }


def report(records, embedder_name, val, fr, diag):
    print(f"\n================  ROUTE A — semantic cliché-distance ({embedder_name})  ================")
    print("\nper-model mean distance-from-cliché (higher = more original) vs judged originality:")
    dist = per_model_mean(records, "distance"); orig = per_model_mean(records, "judged_originality")
    for m in sorted(dist, key=lambda m: -dist[m]):
        o = f"{orig[m]:.2f}" if m in orig else "  — "
        flag = "  <- worst (canary)" if m == WORST else "  <- control (flash)" if m == CONTROL else ""
        print(f"  {m:28} dist={dist[m]:.3f}   judged-orig={o}{flag}")

    print("\n--- STEP 2: VALIDATION kill-switch (run before trusting any frontier result) ---")
    print("  " + val.explain())
    if not val.passes:
        print("  >> STOP: do not interpret the frontier test below; the metric failed validation.")

    print("\n--- STEP 3: FRONTIER discrimination + flash positive control ---")
    sh = lambda m: "gpt5.5" if "gpt-5" in m else m.split("-")[1] if m.startswith("claude") else m
    print("  trio pairs (paired by item):")
    for (a, b), (d, p, n) in fr.trio_pairs.items():
        print(f"    {sh(a):7} vs {sh(b):7}  Δ={d:+.3f}  p={p:.3f}  (n={n} items)")
    print("  flash positive control (each flagship vs gemini-3.5-flash; want Δ>0, p<.05):")
    for (a, b), (d, p, n) in fr.control_pairs.items():
        print(f"    {sh(a):7} vs flash    Δ={d:+.3f}  p={p:.3f}  (n={n} items)")
    print(f"  control resolves known gap: {fr.control_resolves} | trio separates: {fr.trio_separates}")
    print("  => " + fr.branch())

    # --- STEP 5: signal diagnostics (is the weak rank signal real? is flash too close?) ---
    print("\n--- STEP 5: signal diagnostics ---")
    glc = diag["generation_level"]
    if glc:
        print(f"  generation-level (n={glc['n']}, POWERED): "
              f"Pearson {glc['pearson']:+.3f} (p={glc['pearson_p']:.3g}), "
              f"Spearman {glc['spearman']:+.3f} (p={glc['spearman_p']:.3g})  "
              f"<- the real test for 'is there signal'")
    mls = diag["model_level"]
    print(f"  model-level (n={mls['n_models']}): Spearman {mls['rho']:+.3f}, permutation p={mls['perm_p']:.3f}  "
          f"({'significant' if mls['perm_p'] < 0.05 else 'NOT significant — rank trend could be chance'})")

    print("  resolution ladder (trio vs opponents at increasing originality-gap; want big-gap rungs to resolve):")
    for rung in diag["resolution_ladder"]:
        mark = "RESOLVES ✓" if rung["resolved"] else "null ✗"
        print(f"    {rung['rung']:26} judged-gap={rung['judged_orig_gap']:+.2f}  "
              f"Δdist={rung['delta_distance']:+.3f}  p={rung['p']:.3f}  -> {mark}")

    nglc = diag["within_item_normalized"]
    if nglc and glc:
        lifted = abs(nglc["spearman"]) > abs(glc["spearman"]) + 0.05
        verdict = ("LIFTS signal ⇒ item-variance was masking it" if lifted
                   else "no meaningful lift ⇒ item-variance was not the bottleneck")
        print(f"  within-item-normalised generation-level: Spearman {nglc['spearman']:+.3f} "
              f"(p={nglc['spearman_p']:.3g})  vs raw {glc['spearman']:+.3f}  -> {verdict}")


def _write_jsonl_idempotent(path: Path, rows: list[dict], match: dict):
    """Append-style but idempotent per cell: drop existing rows matching `match`
    (e.g. {embedder, reference}), then write the kept rows + the new ones. Re-running
    a cell overwrites just that cell, so the file is always 'current state'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if path.exists():
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if not all(r.get(k) == v for k, v in match.items()):
                kept.append(r)
    path.write_text("\n".join(json.dumps(r) for r in kept + rows) + "\n")


def persist_run(out_dir: Path, embedder_name: str, reference: str, records, val, fr, diag,
                extractor: str | None = None):
    """Persist the per-generation distances (audit trail) and the per-run aggregate
    summary (incl. STEP 5 signal diagnostics) — both idempotent per (embedder, reference,
    extractor), so the full result lives on disk and nothing needs copying out of the
    terminal. `extractor` is None for whole-passage, the extractor name for construal mode."""
    cell = {"embedder": embedder_name, "reference": reference, "extractor": extractor}
    rows = distance_rows(records, embedder_name, reference, extractor)
    _write_jsonl_idempotent(out_dir / "distances.jsonl", rows, cell)

    dist = per_model_mean(records, "distance")
    summary = {**cell, "mode": ("construal" if extractor else "whole-passage"),
               "n_generations": len(records),
               "spearman_vs_judged": val.spearman, "validation_passes": val.passes,
               "worst_model": val.worst_model, "worst_rank_by_distance": val.worst_rank_by_distance,
               "band": (max(dist.values()) - min(dist.values())) if dist else None,
               "per_model_mean_distance": dist,
               "trio_pairs": {f"{a}|{b}": {"delta": d, "p": p, "n": n}
                              for (a, b), (d, p, n) in fr.trio_pairs.items()},
               "control_pairs": {f"{a}|{b}": {"delta": d, "p": p, "n": n}
                                 for (a, b), (d, p, n) in fr.control_pairs.items()},
               "control_resolves": fr.control_resolves, "trio_separates": fr.trio_separates,
               "branch": fr.branch(),
               "signal_diagnostics": diag}   # STEP 5 — gen-level corr, model-level perm-p, ladder, normalised
    _write_jsonl_idempotent(out_dir / "summary.jsonl", [summary], cell)
    return out_dir / "distances.jsonl", out_dir / "summary.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="data/runs/signal")
    ap.add_argument("--embedder", default="stub", choices=["st", "openai", "gemini", "stub"])
    ap.add_argument("--embedder-b", default=None, choices=["st", "openai", "gemini", "stub"],
                    help="second family for the step-4 embedder-as-prior cross-check")
    ap.add_argument("--embedder-model", default=None,
                    help="override the model id for --embedder (e.g. a newer gemini embedding id); "
                         "defaults: gemini→gemini-embedding-2, openai→text-embedding-3-small, st→all-MiniLM-L6-v2")
    ap.add_argument("--embedder-b-model", default=None, help="model id override for --embedder-b")
    ap.add_argument("--reference", default="statement", choices=list(REFERENCE_SETS),
                    help="cliché reference set: 'statement' (abstract) or 'narrative' (register-matched, §11.2 retry)")
    ap.add_argument("--construal-extract", action="store_true",
                    help="THE DILUTION CHECK: extract the object's meaning in story A/B and measure THOSE "
                         "against the clichés, instead of the whole passage (reintroduces a model; §11.2)")
    ap.add_argument("--extractor", default="stub", choices=["stub", "anthropic", "openai", "google", "mock"],
                    help="construal extractor (stub = non-semantic passage-split, wiring only)")
    ap.add_argument("--extractor-model", default=None, help="model id for the construal extractor")
    ap.add_argument("--extractor-b", default=None, choices=["stub", "anthropic", "openai", "google", "mock"],
                    help="second extractor for the cross-extractor guard (construal mode only): "
                         "do two extractors produce the same construal-distances?")
    ap.add_argument("--extractor-b-model", default=None, help="model id for the second extractor")
    ap.add_argument("--out-dir", default="data/runs/route_a",
                    help="persist per-generation distances + per-run summaries here (idempotent per embedder×reference)")
    ap.add_argument("--no-embed-cache", action="store_true",
                    help="disable the content-hash embedding cache (embeddings_cache.jsonl); by default re-runs "
                         "only embed new text")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    reference = REFERENCE_SETS[args.reference]
    gens, judged = load_data(run_dir)
    _log(f"[routeA] {len(gens)} generations, {len(judged)} with judged originality; reference set: {args.reference}")
    if args.embedder == "stub" and args.embedder_b and args.embedder_b != "stub":
        raise SystemExit(
            f"[routeA] --embedder is 'stub' (lexical, non-semantic) but --embedder-b is "
            f"'{args.embedder_b}'. The PRIMARY analysis (steps 2–3) would run on the stub. "
            f"You almost certainly meant:  --embedder {args.embedder_b} --embedder-b <other-family>  "
            f"(e.g. --embedder openai --embedder-b gemini)."
        )
    if args.embedder == "stub":
        _log("[routeA] WARNING: --embedder stub is lexical, NOT semantic — results are for wiring only.")

    if args.extractor_b and not args.construal_extract:
        raise SystemExit("[routeA] --extractor-b is only meaningful with --construal-extract.")

    # Construal-extraction mode: extract (cached) once per extractor, then distances use construals.
    ext_name = None
    construals = None
    if args.construal_extract:
        if args.extractor == "stub":
            _log("[routeA] WARNING: --extractor stub is a non-semantic passage-split — wiring only.")
        extractor = make_extractor(args.extractor, args.extractor_model)
        ext_name = extractor.name
        _log(f"[routeA] construal-extraction ON (extractor={ext_name}) — the §11.2 dilution check")
        construals = extract_construals(extractor, ext_name, gens, out_dir)

    def _distances(e, cons):
        return (construal_distances_for(e, gens, judged, reference, cons)
                if args.construal_extract else distances_for(e, gens, judged, reference))

    cache_path = out_dir / "embeddings_cache.jsonl"
    def _embedder(kind, model):
        e = make_embedder(kind, model)
        return e if args.no_embed_cache else CachingEmbedder(e, cache_path)

    emb = _embedder(args.embedder, args.embedder_model)
    recs, dist_a = _distances(emb, construals)
    val = validate_against_tiers(recs, worst_model=WORST)
    fr = frontier_separation(recs, trio=TRIO, control=CONTROL)
    diag = signal_diagnostics(recs)
    tag = f"{emb.name}, ref={args.reference}" + (f", construal={ext_name}" if ext_name else "")
    report(recs, tag, val, fr, diag)
    dp, sp = persist_run(out_dir, emb.name, args.reference, recs, val, fr, diag, extractor=ext_name)
    _log(f"[routeA] persisted {len(recs)} per-generation rows → {dp}  +  run summary (incl. STEP 5) → {sp}")

    # --- guard 1: two embedding families (holding the extractor fixed) ---
    if args.embedder_b:
        _log(f"[routeA] embedder guard: cross-checking against {args.embedder_b} …")
        emb_b = _embedder(args.embedder_b, args.embedder_b_model)
        recs_b, dist_b = _distances(emb_b, construals)
        val_b = validate_against_tiers(recs_b, worst_model=WORST)
        fr_b = frontier_separation(recs_b, trio=TRIO, control=CONTROL)
        persist_run(out_dir, emb_b.name, args.reference, recs_b, val_b, fr_b, signal_diagnostics(recs_b),
                    extractor=ext_name)
        rho = two_embedder_agreement(dist_a, dist_b)
        print(f"\n--- EMBEDDER-as-prior guard ({emb.name} vs {emb_b.name}) ---")
        print(f"  per-generation distance Spearman = {rho:+.2f}  "
              f"({'robust to embedder choice ✓' if rho >= 0.7 else 'embedder-dependent ✗ — metric tracks idiosyncrasy'})")
        _write_jsonl_idempotent(
            out_dir / "step4_embedder_agreement.jsonl",
            [{"embedder_a": emb.name, "embedder_b": emb_b.name, "reference": args.reference,
              "extractor": ext_name, "spearman": rho}],
            {"embedder_a": emb.name, "embedder_b": emb_b.name, "reference": args.reference, "extractor": ext_name},
        )
        _log(f"[routeA] persisted embedder agreement → {out_dir / 'step4_embedder_agreement.jsonl'}")
    else:
        print("\n--- EMBEDDER guard: skipped (pass --embedder-b <family>) ---")

    # --- guard 2: two extractors (holding the embedder fixed) — construal mode only ---
    if args.construal_extract and args.extractor_b:
        if args.extractor_b == "stub":
            _log("[routeA] WARNING: --extractor-b stub is a non-semantic passage-split — wiring only.")
        extractor_b = make_extractor(args.extractor_b, args.extractor_b_model)
        ext_b_name = extractor_b.name
        _log(f"[routeA] extractor guard: extracting with {ext_b_name} …")
        construals_b = extract_construals(extractor_b, ext_b_name, gens, out_dir)
        recs_xb, dist_xb = _distances(emb, construals_b)        # SAME embedder, extractor B
        val_xb = validate_against_tiers(recs_xb, worst_model=WORST)
        fr_xb = frontier_separation(recs_xb, trio=TRIO, control=CONTROL)
        persist_run(out_dir, emb.name, args.reference, recs_xb, val_xb, fr_xb, signal_diagnostics(recs_xb),
                    extractor=ext_b_name)
        rho_x = two_embedder_agreement(dist_a, dist_xb)         # same generic per-gen Spearman, extractor A vs B
        print(f"\n--- EXTRACTOR guard ({ext_name} vs {ext_b_name}, embedder={emb.name}) ---")
        print(f"  per-generation construal-distance Spearman = {rho_x:+.2f}  "
              f"({'robust to extractor choice ✓' if rho_x >= 0.7 else 'extractor-dependent ✗ — metric tracks the extractor'})")
        _write_jsonl_idempotent(
            out_dir / "cross_extractor_agreement.jsonl",
            [{"embedder": emb.name, "reference": args.reference,
              "extractor_a": ext_name, "extractor_b": ext_b_name, "spearman": rho_x}],
            {"embedder": emb.name, "reference": args.reference, "extractor_a": ext_name, "extractor_b": ext_b_name},
        )
        _log(f"[routeA] persisted extractor agreement → {out_dir / 'cross_extractor_agreement.jsonl'}")
    elif args.construal_extract:
        print("\n--- EXTRACTOR guard: skipped (pass --extractor-b <provider>) ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
