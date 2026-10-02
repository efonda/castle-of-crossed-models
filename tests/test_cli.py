"""Tests for run config + the CLI pipeline stages, driven entirely on the mock
provider (no API key). Exercises the stage functions directly and the full
`run_pipeline`, plus resumability."""

import json
from pathlib import Path

import pytest

from calvino.config import RunConfig, load_run_config
from calvino.cli import (
    export_csv,
    run_pipeline,
    stage_analyze,
    stage_generate,
    stage_judge,
    stage_validate,
)
from calvino.models import Condition, Generation, JudgeScore, Validation
from calvino.storage import GENERATIONS, JUDGE_SCORES, VALIDATIONS, Store

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO / "configs" / "run.example.yaml"

# A fixed single-item fixture so these tests stay decoupled from the production
# item bank (which grows as items are authored) and the counts stay deterministic.
FIXTURE_ITEMS = """\
version: "test-1"
items:
  - item_id: lantern-01
    object: The Lantern
    object_type: tarot_image_object
    description: a battered tin lantern
    story_a: A smuggler waits on the headland.
    story_b: A widow descends into the flooded cellar.
    prompt_variants: [base]
"""


def _config(data_dir) -> RunConfig:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    items_path = data_dir / "items.yaml"
    items_path.write_text(FIXTURE_ITEMS)
    return RunConfig.model_validate({
        "data_dir": str(data_dir),
        "item_bank": str(items_path),
        "samples": 2,
        "bootstrap": 50,
        "models": [
            {"name": "m-a", "provider": "mock", "conditions": ["normal"]},
            {"name": "m-b", "provider": "mock", "conditions": ["normal", "thinking"]},
        ],
        "judges": [
            {"judge_model": "j-x", "provider": "mock"},
            {"judge_model": "j-y", "provider": "mock"},
        ],
    })


# --- config ------------------------------------------------------------------

def test_example_config_loads_and_validates():
    cfg = load_run_config(EXAMPLE_CONFIG)
    assert cfg.data_dir
    assert any(Condition.THINKING in m.conditions for m in cfg.models)  # paired model
    assert len(cfg.judges) == 3


def test_model_spec_defaults():
    cfg = RunConfig.model_validate({
        "data_dir": "x", "item_bank": "y",
        "models": [{"name": "m"}], "judges": [{"judge_model": "j"}],
    })
    assert cfg.models[0].provider == "anthropic"
    assert cfg.models[0].conditions == [Condition.NORMAL]


# --- stages ------------------------------------------------------------------

def test_stage_generate_counts_and_persists(tmp_path):
    cfg = _config(tmp_path)
    result = stage_generate(cfg)
    # 1 item x (m-a: 1 cond + m-b: 2 conds) x 2 samples = 6
    assert result["new_generations"] == 6
    assert len(Store(tmp_path).load(GENERATIONS, Generation)) == 6


def test_stage_generate_is_resumable(tmp_path):
    cfg = _config(tmp_path)
    stage_generate(cfg)
    again = stage_generate(cfg)
    assert again["new_generations"] == 0  # nothing new on re-run


def test_stage_validate_runs_once_per_generation(tmp_path):
    cfg = _config(tmp_path)
    stage_generate(cfg)
    assert stage_validate(cfg)["new_validations"] == 6
    assert stage_validate(cfg)["new_validations"] == 0  # resumable
    assert len(Store(tmp_path).load(VALIDATIONS, Validation)) == 6


def test_stage_judge_scores_every_pair(tmp_path):
    cfg = _config(tmp_path)
    stage_generate(cfg)
    result = stage_judge(cfg)
    assert result["new_scores"] == 6 * 2  # 6 generations x 2 judges
    assert len(Store(tmp_path).load(JUDGE_SCORES, JudgeScore)) == 12


def test_stage_analyze_writes_summary(tmp_path):
    cfg = _config(tmp_path)
    stage_generate(cfg)
    stage_judge(cfg)
    summary = stage_analyze(cfg)
    assert summary["n_generations"] == 6
    assert "ranking" in summary
    assert (tmp_path / "summary.json").exists()
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["n_scores"] == 12


def test_analyze_with_no_scores_reports_error(tmp_path):
    cfg = _config(tmp_path)
    stage_generate(cfg)  # generations but no judging
    summary = stage_analyze(cfg, write=False)
    assert "error" in summary


def test_judge_summary_separates_cached_from_failures(tmp_path):
    cfg = _config(tmp_path)
    stage_generate(cfg)
    first = stage_judge(cfg)
    # First pass: all fresh, none cached, none failed (mock returns valid JSON).
    assert first["new_scores"] == 12
    assert first["cached"] == 0 and first["parse_failures"] == 0
    # Second pass: everything already scored -> all cached, still zero failures.
    second = stage_judge(cfg)
    assert second["new_scores"] == 0
    assert second["cached"] == 12 and second["parse_failures"] == 0


def test_export_csv_one_row_per_score(tmp_path):
    import csv

    cfg = _config(tmp_path)
    stage_generate(cfg)   # 6 generations
    stage_judge(cfg)      # 2 judges -> 12 scores
    result = export_csv(cfg)

    assert result["rows"] == 12
    with open(result["path"], newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 12
    # core columns present and populated
    for col in ("model", "judge_model", "primary", "output_text", "evidence_bridge", "spans_verified"):
        assert col in reader.fieldnames
    assert rows[0]["spans_verified"].endswith("/5")


def test_run_pipeline_end_to_end(tmp_path):
    cfg = _config(tmp_path)
    summaries = run_pipeline(cfg)
    stages = [s["stage"] for s in summaries]
    assert stages == ["generate", "validate", "judge", "analyze"]
    assert summaries[0]["new_generations"] == 6
    assert summaries[2]["new_scores"] == 12
    assert summaries[3]["n_generations"] == 6
    # The judge evidence spans (from the mock) verify against the canned prose.
    scores = Store(tmp_path).load(JUDGE_SCORES, JudgeScore)
    assert all(all(s.spans_verified.values()) for s in scores)
