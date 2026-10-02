"""Live probe: do the judges reason under the NORMAL condition? CALLS MODEL APIS.

Sends one small scoring request per judge provider under the NORMAL condition and reports the
reasoning tokens each provider returns. Persisted judge records carry no token usage, so this is
the only way to check a judge's reasoning state. Not needed to reproduce any number in the paper.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")
from calvino.models import Condition  # noqa: E402
from calvino.providers import GenerationRequest, get_provider  # noqa: E402

# A miniature scoring prompt of realistic shape: the real one is longer, but reasoning-by-default
# is a per-call policy, not a function of prompt length.
PROMPT = (
    "Score this four-sentence passage 0-5 on distinctness and quote a verbatim span for each "
    'score. Output ONLY a JSON object with keys distinctness_score, distinctness_evidence.\n\n'
    "PASSAGE: The lantern swung at the prow and turned the fog into a pale road. He held it "
    "steady until the patrol turned back. At dawn the widow found it wedged in her flooded "
    "cellar. She turned it in her hands, deciding whether it had been her husband's."
)

# (label, provider name, model id) — the judge roster, new judges first.
JUDGES = [
    ("grok-4.5            (NEW, xAI)", "xai", "grok-4.5"),
    ("gemini-3.6-flash    (NEW, Google)", "google", "gemini-3.6-flash"),
    ("grok-4.3            (published, xAI)", "xai", "grok-4.3"),
    ("gemini-3.1-pro-prev (published, Google)", "google", "gemini-3.1-pro-preview"),
    ("qwen3.7-max         (published, Alibaba)", "openrouter", "qwen/qwen3.7-max"),
]

KEY_FOR = {"xai": ["XAI_API_KEY"], "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
           "openrouter": ["OPENROUTER_API_KEY"]}

# Usage fields that mean "the model thought", across the three SDK shapes.
THINK_KEYS = ("thoughts_token_count", "reasoning_tokens", "reasoning_token_count",
              "completion_tokens_details", "output_tokens_details")


def cost_fields(usage: dict) -> str:
    """xAI returns `cost_in_usd_ticks` and OpenRouter returns `cost` in the usage block, so for
    those two vendors the price of a judge call is measurable rather than estimated. Google
    reports no cost field. Printed raw (no unit guessing) plus a x198 extrapolation to the
    sibling-seam run size, so whatever the tick scale is, the run total is one multiplication."""
    out = []
    for k in ("cost", "cost_in_usd_ticks", "total_cost"):
        v = usage.get(k)
        if isinstance(v, (int, float)):
            out.append(f"{k}={v!r}  (x198 calls = {v * 198:,.4f})")
    d = usage.get("cost_details")
    if isinstance(d, dict) and d:
        out.append(f"cost_details={d}")
    return "; ".join(out) or "no cost field reported by this provider"


def io_tokens(usage: dict) -> str:
    """Input/output token counts, across the three SDK naming conventions."""
    def pick(*names):
        for n in names:
            v = usage.get(n)
            if isinstance(v, int):
                return v
        return None
    i = pick("input_tokens", "prompt_tokens", "prompt_token_count")
    o = pick("output_tokens", "completion_tokens", "candidates_token_count")
    return f"in={i} out={o}"


def thinking_tokens(usage: dict) -> int | None:
    """Pull a reasoning-token count out of whatever shape the SDK returned."""
    for k in ("thoughts_token_count", "reasoning_tokens", "reasoning_token_count"):
        v = usage.get(k)
        if isinstance(v, int):
            return v
    for k in ("completion_tokens_details", "output_tokens_details"):
        d = usage.get(k)
        if isinstance(d, dict):
            for kk in ("reasoning_tokens", "reasoning_token_count"):
                if isinstance(d.get(kk), int):
                    return d[kk]
    return None


def main() -> int:
    print("Probing the judge call path (condition=NORMAL, temperature=0.0) for reasoning tokens.\n")
    rows: list[tuple[str, str]] = []
    for label, provider_name, model in JUDGES:
        keys = KEY_FOR[provider_name]
        if not any(os.environ.get(k) for k in keys):
            print(f"{label:42s} SKIPPED ({'/'.join(keys)} unset)")
            continue
        try:
            provider = get_provider(provider_name)
            result = provider.generate(GenerationRequest(
                prompt=PROMPT, model=model, condition=Condition.NORMAL,
                temperature=0.0, max_tokens=1024, json_schema={"type": "object"},
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"{label:42s} ERROR  {str(exc)[:100]}")
            rows.append((label, "error"))
            continue

        usage = (result.provider_meta or {}).get("usage") or {}
        tt = thinking_tokens(usage)
        verdict = ("REASONED" if tt else "no reasoning tokens") if tt is not None else "NOT REPORTED"
        if tt:
            verdict += f" ({tt} tok)"
        print(f"{label:42s} {verdict}")
        print(f"{'':42s} tokens: {io_tokens(usage)}")
        print(f"{'':42s} cost:   {cost_fields(usage)}")
        if any(k in usage for k in THINK_KEYS):
            print(f"{'':42s} {{{', '.join(f'{k}={usage[k]!r}' for k in THINK_KEYS if k in usage)}}}")
        rows.append((label, verdict))

    if not rows:
        print("\nNo probes ran (no API keys set).", file=sys.stderr)
        return 1

    print("\nSummary for the appendix (judge reasoning mode, MEASURED):")
    for label, verdict in rows:
        print(f"  {label:42s} {verdict}")
    print("\nNote: a judge that reasons is not a flaw -- it is uniform across every contestant that")
    print("judge scores, so it cannot favour one model over another. What matters is (a) that the")
    print("paper states it accurately, and (b) that the two members of a same-lab PAIR ran the same")
    print("way, since the sibling-seam measurement compares them to each other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
