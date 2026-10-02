# The Castle of Crossed Models

Code and data for **"The Castle of Crossed Models: Benchmarking LLMs on Calvinian Narrative
Crossings"** (Enrico Fonda, AI Measurement Science Workshop at COLM 2026).

The benchmark asks a model for one four-sentence passage in which two stories cross through a
single object that means something different in each story. Format is checked by code; quality
is scored by blinded LLM judges from several labs, using absolute rubric scores and pairwise
comparison. The same data that ranks the models is used to measure the judge panel itself:
reliability, same-lab bias, and where the rubric saturates.

This release (`v1.0-aims`) contains everything needed to reproduce the paper's numbers from the
persisted data, without any API calls.

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                      # 153 tests, no API keys needed
```

Run the whole pipeline (generate → validate → blind judge → analyze) on mock providers, with no
API key:

```bash
uv run python main.py run --config configs/run.example.yaml
```

## Data

`data/runs/<run>/` holds append-only JSONL files:

| file | content |
|---|---|
| `generations.jsonl` | one record per generation: item, model, condition, sample, temperature, seed, the rendered prompt verbatim, the output text, provider metadata (including token usage) |
| `validations.jsonl` | deterministic format checks: sentence count, two-story split, object presence in each half |
| `judge_scores.jsonl` | parsed absolute scores per (generation, judge), evidence spans and their verification |
| `raw_judge_responses.jsonl` | every judge reply verbatim, written before parsing |
| `pairwise_votes.jsonl` | pairwise judgments, both presentation orders |

A generation's id is the hash of its request, so re-running a config skips completed work.

| run | role in the paper |
|---|---|
| `signal` | 6-item signal bank: headline absolute panel, full-roster ranking, pairwise pool, 0–10 scale, reasoning-mode judges |
| `hard` | 12-item hard bank: absolute scores and pairwise tier calibration |
| `italian` | Italian pilot (App. E) |
| `signal_gemini`, `signal_grok`, `signal_qwen` | additional judges on the signal generations (lab-balanced check, same-lab bonus fit) |
| `sibling-seam-google`, `sibling-seam-xai` | second judge per lab for the same-lab agreement check (App. G) |
| `hard_frontier`, `hard_grok_bo`, `cluster-gemini-hard`, `cluster-gemini-signal` | neutral-lab pairwise adjudication (§3, App. D) |
| `mini-reason-control-signal` | gpt-5.5 with reasoning off (Table 2, bottom) |
| `signal_fable`, `fable-fc`, `hard_fable_fc` | claude-fable-5, absolute and pairwise (§3, App. B, App. G) |
| `mini-fable-effort`, `mini-fable-effort-hard`, `mini-fable-elow-fc` | fable at low effort (reasoning-token measurements, App. G) |
| `item-validation` | out-of-lab item audit (§8) |
| `route_a` | judge-free embedding originality metric (§6); the embeddings cache is omitted and is rebuilt on a re-run |

Item banks: `data/items.yaml` (signal), `data/items_hard.yaml`, `data/items_it.yaml`,
`data/items_it_campana.yaml` (Italian resample). Audit log: `data/item_validation_log.csv`.

## Reproducing the paper

All scripts below read the persisted data and make no API calls. Run them from the repository
root with `uv run python scripts/<name>.py`.

| result | script |
|---|---|
| Same-lab bonus, pooled +0.080 and per judge (§5) | `verify_self_preference.py` |
| Composition dependence of the frontier gap, Δ0.116 / 0.306 / 0.021 (§3) | `verify_aims_deconfound_cuts.py` |
| Same-lab agreement, ICC 0.92 vs 0.83, and the sibling-pair check (§5, App. G) | `icc_seam_bootstrap.py`, `sibling_seam.py` |
| Pre-registered adjudication, 42–5 and 32–8, item sign tests (§3, Table 2, App. D) | `cluster_fc_three_lineages.py` |
| Reasoning-off control (Table 2, bottom) | `verify_reasoning_control.py` |
| Fable pairwise, 54–6 and 117–16 (§3, App. G) | `verify_fable_fc.py` |
| Pairwise tier calibration and within-trio pairwise comparisons (§3) | `trio_crosslab_forcedchoice.py`, `fc_propensity.py` |
| Per-dimension η² and F-tests (§4, Table 3, App. B) | `dimension_discrimination.py` |
| Rank stability ρ (§3) | `verify_rho_calibration.py` |
| Hard-bank saturation and absolute checks (§3, App. B) | `saturation_hard_robustness.py`, `verify_hard_absolute.py` |
| Reasoning tokens per model, e.g. gpt-5.5 696 and fable 335 on the hard bank (§2, App. D, App. G) | `measure_thinking_tokens.py data/runs/hard data/runs/hard_fable_fc` |
| Ranking table (App. B) | `plot_ranking.py` (prints the table without matplotlib) |
| Ranking figure (Fig. 1) | `uv run --with seaborn python scripts/plot_ranking_seaborn.py --out ranking-wide.pdf --drop llama3.2:3b --xmin 3 --figsize 8.0,3.2` |

Not yet scripted: the lab-balanced panel swap (App. B, ρ 0.98 / 0.93) and the Italian-pilot summaries (App. E). Both are computable from the included `signal`, `signal_gemini`, `hard` and `italian` runs.

Scripts that call model APIs, and so need keys and incur cost: `validate_items.py` (item audit),
`route_a_originality.py` (embedding metric), `diagnose_judge_reasoning.py` (a live probe),
and `main.py` with any non-mock config.

## Re-running with live models

Configs in `configs/` record every run's models, judges, temperatures and seeds. Live runs need
the relevant keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `XAI_API_KEY`, and
`OPENROUTER_API_KEY` for qwen. Local models run through [Ollama](https://ollama.com). Re-running a
config against the included data skips every completed call. New generations will not match the
published ones exactly: provider models and defaults change over time, and the paper's results
are a snapshot of June–July 2026 deployments.

## Not included

- Human-study materials, which are governed by participant consent terms.
- A held-out item bank reserved for future confirmatory analysis.
- Material from follow-up work not reported in this paper.

## Licence

Code: MIT ([LICENSE](LICENSE)). Data: CC BY 4.0 ([data/LICENSE](data/LICENSE)). Model outputs in `data/runs/` were generated through the APIs of the
providers named in each record (Anthropic, OpenAI, Google, xAI, Alibaba via OpenRouter) and local
open-weight models; their use remains subject to those providers' terms.

## Citation

```bibtex
@inproceedings{fonda2026castle,
  title     = {The Castle of Crossed Models: Benchmarking {LLMs} on Calvinian Narrative Crossings},
  author    = {Fonda, Enrico},
  booktitle = {AI Measurement Science Workshop at COLM 2026},
  year      = {2026}
}
```
