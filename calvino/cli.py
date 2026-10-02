"""Config-driven CLI for the four pipeline stages plus an all-in-one `run`.

    uv run python main.py generate --config configs/run.example.yaml
    uv run python main.py validate --config ...
    uv run python main.py judge    --config ...
    uv run python main.py analyze  --config ...
    uv run python main.py run      --config ...   # all four in order

Each stage is a plain function taking a RunConfig and returning a small summary
dict, so the whole pipeline is testable without the argv layer. All stages are
resumable: re-running skips work already on disk (no API calls).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import analysis as A
from .config import RunConfig, load_run_config
from .harness import GenerationHarness, expand_specs
from .items import load_item_bank
from .judge import Judge, JudgeOrchestrator, build_blind_set
from .models import Generation, JudgeScore, Validation
from .providers import get_provider
from .storage import GENERATIONS, JUDGE_SCORES, VALIDATIONS, Store
from .validator import validate_generation


def _log(msg: str) -> None:
    """Progress line to stderr (keeps stdout clean for the JSON summary; visible
    live in a terminal and tailable for background runs)."""
    print(msg, file=sys.stderr, flush=True)


def _summarize_timing(stats: dict[str, dict], produced_key: str) -> dict:
    """Per-model/judge timing for the JSON summary: calls, produced, total seconds,
    and seconds-per-call. Wall time per item is attributed to the active model
    (judges/models run sequentially, so this is accurate to within one call)."""
    out = {}
    for name, st in stats.items():
        calls = st["calls"]
        out[name] = {
            "calls": calls,
            produced_key: st[produced_key],
            "seconds": round(st["seconds"], 1),
            "sec_per_call": round(st["seconds"] / calls, 2) if calls else None,
        }
    return out


def _timing_lines(stage: str, stats: dict[str, dict], produced_key: str) -> list[str]:
    lines = []
    for name, st in stats.items():
        calls = st["calls"]
        avg = st["seconds"] / calls if calls else 0.0
        lines.append(f"[{stage}] timing  {name}: {st[produced_key]}/{calls} {produced_key} "
                     f"in {st['seconds']:.1f}s ({avg:.1f}s/call)")
    return lines


def stage_generate(config: RunConfig) -> dict:
    """Fan every item out across models/conditions/variants/samples and generate,
    skipping anything already on disk. Each model uses its configured provider."""
    store = Store(config.data_dir)
    bank = load_item_bank(config.item_bank)

    # Build all specs up front (grouped by provider) so we can show done/total.
    plan = []  # (provider_name, specs)
    for spec in config.models:
        model_specs = []
        for item in bank.items:
            model_specs.extend(expand_specs(
                item, models=[spec.name], samples=config.samples,
                conditions=spec.conditions, temperature=config.temperature,
                api_model=spec.model,  # on-disk name = spec.name; call spec.model if it differs
                effort=spec.effort,    # adaptive-thinking depth for a THINKING condition
            ))
        plan.append((spec.provider, model_specs))
    total = sum(len(s) for _, s in plan)

    stats: dict[str, dict] = {}
    state = {"done": 0, "prev": time.monotonic()}

    def progress(spec, generation):
        now = time.monotonic()
        st = stats.setdefault(spec.model, {"calls": 0, "new": 0, "seconds": 0.0})
        st["calls"] += 1
        st["seconds"] += now - state["prev"]
        if generation is not None:
            st["new"] += 1
        state["prev"] = now
        state["done"] += 1
        tag = "new" if generation is not None else "cached"
        _log(f"[generate] {state['done']}/{total}  {spec.model}  {spec.item.item_id} "
             f"s{spec.sample_idx} {spec.condition.value}  -> {tag}")

    _log(f"[generate] starting: {total} specs across {len(plan)} model(s)")
    produced = 0
    for provider_name, model_specs in plan:
        harness = GenerationHarness(store, get_provider(provider_name))
        produced += len(harness.run_all(model_specs, progress=progress))

    timing = _summarize_timing(stats, "new")
    for line in _timing_lines("generate", stats, "new"):
        _log(line)
    _log(f"[generate] done: {produced} new, {total - produced} cached")
    return {"stage": "generate", "new_generations": produced, "timing": timing}


def stage_validate(config: RunConfig) -> dict:
    """Run the deterministic validator over every generation not yet validated."""
    store = Store(config.data_dir)
    done = store.validated_ids()
    produced = 0
    for gen in store.iter(GENERATIONS, Generation):
        if gen.id in done:
            continue
        store.append(VALIDATIONS, validate_generation(gen))
        produced += 1
    _log(f"[validate] {produced} new validations")
    return {"stage": "validate", "new_validations": produced}


def stage_judge(config: RunConfig) -> dict:
    """Blind, randomise, and score every generation with every configured judge."""
    store = Store(config.data_dir)
    generations = store.load(GENERATIONS, Generation)
    blind_set = build_blind_set(generations, seed=config.blind_seed)
    judges = [
        Judge(j.judge_model, get_provider(j.provider), api_model=j.model,
              thinking=j.thinking, effort=j.effort, thinking_budget=j.thinking_budget,
              scale=j.scale, language=j.language, execution=j.execution)
        for j in config.judges
    ]

    total = len(judges) * len(blind_set)
    pre_judged = store.judged_pairs()  # snapshot before the run: these are cache hits, not failures
    stats: dict[str, dict] = {}
    totals = {"cached": 0, "failed": 0}
    state = {"done": 0, "prev": time.monotonic()}

    def progress(judge, blind, score):
        now = time.monotonic()
        st = stats.setdefault(judge.judge_model, {"calls": 0, "scored": 0, "seconds": 0.0})
        st["calls"] += 1
        st["seconds"] += now - state["prev"]
        if score is not None:
            st["scored"] += 1
            tag = "scored"
        elif (blind.generation_id, judge.judge_model) in pre_judged:
            totals["cached"] += 1
            tag = "cached"  # already on disk — skipped, no API call
        else:
            totals["failed"] += 1
            tag = "PARSE-FAIL"  # genuine: API returned but JSON didn't parse (raw saved)
        state["prev"] = now
        state["done"] += 1
        _log(f"[judge] {state['done']}/{total}  {judge.judge_model}  {blind.blind_id}  -> {tag}")

    totals["quota_skipped"] = 0

    def on_quota(judge, exc):
        totals["quota_skipped"] += 1
        _log(f"[judge] QUOTA hit for {judge.judge_model} — skipping its remaining outputs; "
             f"re-run later to resume ({exc})")

    _log(f"[judge] starting: {len(judges)} judge(s) x {len(blind_set)} outputs = {total} calls")
    produced = JudgeOrchestrator(store).run(blind_set, judges, progress=progress, on_quota=on_quota)
    if totals["quota_skipped"]:
        _log(f"[judge] {totals['quota_skipped']} judge(s) skipped on quota — re-run the same "
             f"command after the quota resets to fill the gaps (cache resumes automatically)")

    timing = _summarize_timing(stats, "scored")
    for line in _timing_lines("judge", stats, "scored"):
        _log(line)
    if totals["failed"]:
        _log(f"[judge] WARNING: {totals['failed']} genuine parse failures (raw responses saved)")
    _log(f"[judge] done: {len(produced)} new scores, {totals['cached']} cached, {totals['failed']} parse-fail")
    return {"stage": "judge", "new_scores": len(produced),
            "cached": totals["cached"], "parse_failures": totals["failed"], "timing": timing}


def export_csv(config: RunConfig, *, out: str | None = None) -> dict:
    """Write a tidy CSV — one row per (generation x judge score) — joining
    generation metadata, the five dimension scores + primary aggregate, structural
    gate fields, span verification, the output text, and the judge's evidence spans.
    Reviews cleanly in a spreadsheet / pivots by model or judge."""
    import csv

    store = Store(config.data_dir)
    gens = {g.id: g for g in store.iter(GENERATIONS, Generation)}
    vals = {v.generation_id: v for v in store.iter(VALIDATIONS, Validation)}
    dims = list(A.ALL_DIMS)

    rows = []
    for s in store.iter(JUDGE_SCORES, JudgeScore):
        g = gens.get(s.generation_id)
        if g is None:
            continue
        v = vals.get(s.generation_id)
        sc = s.scores
        row = {
            "model": g.model,
            "item_id": g.item_id,
            "object_type": g.object_type.value,
            "condition": g.condition.value,
            "prompt_variant": g.prompt_variant.value,
            "sample_idx": g.sample_idx,
            "judge_model": s.judge_model,
            "distinctness": sc.distinctness,
            "bridge": sc.bridge,
            "tonal": sc.tonal,
            "originality": sc.originality,
            "combinatorial": sc.combinatorial,
            "primary": round((sc.distinctness + sc.bridge + sc.tonal + sc.originality) / 4, 3),
            "sentence_count": v.sentence_count if v else None,
            "count_ok": v.count_ok if v else None,
            "object_in_a": v.object_in_a if v else None,
            "object_in_b": v.object_in_b if v else None,
            "spans_verified": f"{sum(s.spans_verified.values())}/{len(dims)}" if s.spans_verified else "",
            "generation_id": g.id,
            "blind_id": s.blind_id,
            "output_text": g.output_text,
        }
        for d in dims:
            row[f"evidence_{d}"] = getattr(s.evidence, d)
        rows.append(row)

    rows.sort(key=lambda r: (r["item_id"], r["model"], r["sample_idx"], r["judge_model"]))
    out_path = Path(out) if out else Path(config.data_dir) / "scores.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _log(f"[export] wrote {len(rows)} rows to {out_path}")
    return {"stage": "export", "rows": len(rows), "path": str(out_path)}


def export_model_summary(config: RunConfig, *, out: str | None = None) -> dict:
    """Write a simplified per-model table: mean of each dimension + primary,
    averaged across judges and items/samples. Uses only judges with complete
    coverage of all generations (for a balanced average); falls back to all judges
    if none are complete. Also prints the table."""
    store = Store(config.data_dir)
    df = A.build_scores_frame(store)
    if df.empty:
        return {"stage": "export-summary", "error": "no judge scores on disk"}

    n_gen = df["generation_id"].nunique()
    coverage = df.groupby("judge_model")["generation_id"].nunique()
    used = [j for j, c in coverage.items() if c == n_gen] or list(coverage.index)
    sub = df[df["judge_model"].isin(used)].assign(primary=lambda d: A.primary_aggregate(d))

    summary = (
        sub.groupby("model")
        .agg(
            n_scores=("primary", "size"),
            distinctness=("distinctness", "mean"),
            bridge=("bridge", "mean"),
            tonal=("tonal", "mean"),
            originality=("originality", "mean"),
            combinatorial=("combinatorial", "mean"),
            primary=("primary", "mean"),
        )
        .round(2)
        .sort_values("primary", ascending=False)
        .reset_index()
    )

    out_path = Path(out) if out else Path(config.data_dir) / "model_summary.csv"
    summary.to_csv(out_path, index=False)
    _log(f"[export] per-model summary; judges averaged (complete coverage): {used}")
    _log("\n" + summary.to_string(index=False))
    return {"stage": "export-summary", "models": int(len(summary)),
            "judges_used": used, "path": str(out_path)}


def stage_analyze(config: RunConfig, *, write: bool = True) -> dict:
    """Load scores from disk and run the Phase 1 analyses. Writes a summary.json
    to the data dir and returns the same summary."""
    store = Store(config.data_dir)
    df = A.build_scores_frame(store)
    if df.empty:
        return {"stage": "analyze", "error": "no judge scores on disk"}

    orth = A.combinatorial_orthogonality(df)
    rank = A.rank_stability(df, n_boot=config.bootstrap, seed=config.blind_seed)
    frontier = A.frontier_separation(df, n_boot=config.bootstrap, seed=config.blind_seed)

    summary = {
        "stage": "analyze",
        "n_scores": int(len(df)),
        "n_generations": int(df["generation_id"].nunique()),
        "descriptive_means": A.descriptive_means(df).to_dict(orient="records"),
        "judge_agreement_icc": A.judge_agreement(df).to_dict(orient="records"),
        "ranking": rank.ranking,
        "rank_mean_spearman": rank.mean_spearman,
        "frontier_table": frontier.table.to_dict(orient="records"),
        "frontier_top_saturated": frontier.top_saturated,
        "combinatorial_orthogonality": {"r": orth.r, "p": orth.p, "n": orth.n},
        "thinking_effect": A.thinking_effect(df).to_dict(orient="records"),
        "failure_worst": A.failure_catalogue(df)["worst"].to_dict(orient="records"),
    }
    if write:
        out = Path(config.data_dir) / "summary.json"
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summary["summary_path"] = str(out)
    return summary


def run_pipeline(config: RunConfig) -> list[dict]:
    """Run all four stages in order. Returns each stage's summary."""
    return [
        stage_generate(config),
        stage_validate(config),
        stage_judge(config),
        stage_analyze(config),
    ]


_STAGES = {
    "generate": stage_generate,
    "validate": stage_validate,
    "judge": stage_judge,
    "analyze": stage_analyze,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calvino", description="Calvino Benchmark pilot runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (*_STAGES, "run"):
        p = sub.add_parser(name, help=f"{name} stage")
        p.add_argument("--config", required=True, help="path to run config YAML")
    p_export = sub.add_parser("export", help="export judge scores to CSV")
    p_export.add_argument("--config", required=True, help="path to run config YAML")
    p_export.add_argument("--out", default=None, help="CSV path (default: <data_dir>/scores.csv)")
    p_export.add_argument("--by-model", action="store_true",
                          help="simplified per-model summary (averaged across judges + items)")

    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    if args.command == "run":
        for summary in run_pipeline(config):
            print(json.dumps(summary, default=str))
    elif args.command == "export":
        fn = export_model_summary if args.by_model else export_csv
        print(json.dumps(fn(config, out=args.out), indent=2, default=str))
    else:
        print(json.dumps(_STAGES[args.command](config), indent=2, default=str))
    return 0
