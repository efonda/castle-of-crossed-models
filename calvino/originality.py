"""Route A — an objective, judge-free originality metric (technical report §11.2).

The hypothesis: a *more original* crossing sits farther (in semantic space) from the
**predictable, stale construals** of its object. We measure that distance and ask
whether it (a) validates against the judges' originality scores and (b) separates the
flagship trio where the LLM judges could not — all with **no judge in the loop**.

This module is deliberately model-free at its core: it holds the hand-authored cliché
reference set, the distance metric, and the four kill-switch-ordered analysis steps as
pure functions over embedding arrays. The embedder itself is injected (see
`scripts/route_a_originality.py`), so the *same* logic runs under two different
embedding families — which is the empirical guard against tracking one embedder's
idiosyncrasies (§11.2 step 4).

The pilot that ruled out the *surface* (keyword / lexical-novelty) version of this
metric is in §11.2; this is the semantic build it narrowed Route A to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Step 1 — the cliché reference set (the load-bearing design decision, §11.2).
#
# For each object: the OBVIOUS / stale construals — the first reading anyone reaches
# for. Authored as short *sentences* (not bare keywords) because they are embedded
# semantically: a flagship can commit the "scales → justice" cliché without the token
# "justice", and only a semantic reference can catch that (the keyword pilot could not).
#
# MULTI-SOURCE by design (§11.2 caution): these are hand-authored seeds. Append
# weak-model-authored construals via `extra_reference=` in the runner so the set is not
# one source's idea of "obvious" — a single-source reference reintroduces the §9
# single-prior problem. A bad/single-source reference set is the one thing no embedder
# can rescue.
# ---------------------------------------------------------------------------
CLICHE_REFERENCE: dict[str, list[str]] = {
    "lantern-01": [
        "The lantern is a symbol of hope, guiding the lost through the darkness.",
        "Like a beacon of faith, the lantern lights the way home.",
        "The lantern represents the soul's enduring light against the dark.",
        "A guiding light in the night, the lantern shows the path forward.",
    ],
    "tower-01": [
        "The tower stands for human pride brought low, like the Tower of Babel.",
        "The tower is a symbol of hubris and inevitable downfall.",
        "An ivory tower of ambition, struck down by fate.",
        "The tower represents isolation and the fall of the mighty.",
    ],
    "wheel-01": [
        "The wheel is the wheel of fortune, turning fate from high to low.",
        "The wheel symbolizes the endless cycle of destiny and karma.",
        "Ever-turning, the wheel shows that fortune is fickle.",
        "The wheel represents the inexorable cycle of time and fate.",
    ],
    "scales-01": [
        "The scales are the scales of justice, weighing right against wrong.",
        "The scales symbolize balance, fairness, and the rule of law.",
        "Like Lady Justice, the scales weigh the souls of the guilty.",
        "The scales represent moral equilibrium and impartial judgment.",
    ],
    # Invented object — weak canonical cliché, by design (no settled symbolism). Fewer
    # references; expect higher baseline distance for everyone (already the most
    # original-eliciting item in §4.4). Kept so the metric is defined for every item.
    "salt-compass-01": [
        "The compass points the way, a symbol of finding true north in life.",
        "Salt preserves and the compass guides: direction and endurance.",
        "The salt-compass represents finding one's bearings when lost at sea.",
    ],
    "key-01": [
        "The key unlocks hidden secrets and opens the door to mystery.",
        "The key is a symbol of access, the answer that unlocks the truth.",
        "A key to freedom, opening what was locked away.",
        "The key represents the solution that unlocks a new beginning.",
    ],
    # --- Italian arm (DRAFT — ONE model-authored source; native-review + augment) -----
    # Seeds, not the validated set: the reference set is multi-source by design (header).
    # Native-REVIEW these (generative cultural-cliché calibration is the load-bearing risk),
    # then augment with native + weak-model seeds (`extra_reference=`). Keyed to items_it.yaml
    # (it-0.2.0-draft) — must track it. Cliche-trap items have strong canonical readings;
    # invented items have only weak stock readings by design (fewer references).
    "bilancia-01": [  # cliche-trap: la bilancia della giustizia — overwhelmingly canonical in IT
        "La bilancia è la bilancia della giustizia, che pesa il bene contro il male.",
        "La bilancia simboleggia l'equilibrio, l'equità e l'imperio della legge.",
        "Come la Giustizia bendata, la bilancia pesa le anime dei colpevoli.",
        "La bilancia rappresenta l'equilibrio morale e il giudizio imparziale.",
    ],
    "specchietto-01": [  # cliche-trap: lo specchio → vanità / verità / doppio
        "Lo specchio è il simbolo della vanità e dell'apparenza che inganna.",
        "Specchio dell'anima, riflette la verità di ciò che siamo davvero.",
        "Lo specchio mostra il nostro doppio, l'altro nascosto dentro di noi.",
        "Nello specchio la verità nuda si rivela a chi osa guardarsi.",
    ],
    "campana-01": [  # cliche-trap: la campana → morte / fede / il tempo del paese
        "La campana suona a morto, annuncia la fine e il lutto del paese.",
        "La campana chiama i fedeli alla preghiera e alla fede.",
        "Per chi suona la campana: il rintocco che segna il destino di tutti.",
        "La campana scandisce il tempo sacro del villaggio e delle generazioni.",
    ],
    "bambola-pane-01": [  # invented: weak stock readings only (pane→vita; bambola→infanzia)
        "Il pane è il simbolo della vita e del nutrimento condiviso.",
        "Il pane spezzato è sacrificio e comunione.",
        "La bambola è l'innocenza dell'infanzia perduta.",
    ],
    "gabbia-sasso-01": [  # invented: weak stock reading (gabbia→prigionia/libertà)
        "La gabbia è il simbolo della prigionia e della libertà negata.",
        "L'uccello in gabbia è l'anima che sogna di volare via.",
        "La gabbia vuota parla di libertà perduta o ritrovata.",
    ],
    "ombrello-cucito-01": [  # invented: weak stock reading (ombrello→protezione/riparo)
        "L'ombrello è il simbolo della protezione e del riparo dalle tempeste.",
        "Aprire l'ombrello: ripararsi dalle avversità della vita.",
        "L'ombrello chiuso è la protezione negata, l'esposizione al mondo.",
    ],
}

# Register-matched variant (§11.2 retry). The first reference set states the cliché as an
# ABSTRACT symbol-definition; against concrete 4-sentence narrative generations, the
# dominant axis of an embedding comparison is genre/register, not construal — which
# compressed every model into a razor-thin distance band (no resolution; flash control
# null). This set writes the SAME stale construals as short NARRATIVE SCENES in the
# generations' own register, so distance reflects "did it reach for the obvious story"
# rather than "is this prose or a definition." Select with `--reference narrative`.
CLICHE_REFERENCE_NARRATIVE: dict[str, list[str]] = {
    "lantern-01": [
        "Lost in the howling storm, the sailor saw the lantern's steady glow on the far shore. Hope flooded back into his chest. He fixed his eyes on that small light and rowed toward it. The lantern guided him safely home through the dark.",
        "The old woman kept a lantern burning in the window every night. It was her faith made visible, a beacon for her son still lost at sea. As long as it glowed, she believed, he could find his way back. The light was her hope and her prayer.",
        "In the blackness of the mine, the boy lifted the lantern high. Its warm light pushed back the dark and the fear. It showed him the path out, step by trembling step. The lantern was the one small hope he carried.",
    ],
    "tower-01": [
        "The king raised his tower higher each year, certain it would touch heaven itself. His pride grew with every stone. But such hubris cannot stand, and like Babel the tower was thrown down. Its ruin taught that the mighty always fall.",
        "From his lonely tower the sorcerer looked down on the world he scorned. He had shut himself away in pride, above all other men. In the end his ambition isolated him utterly. The tower became his prison and his tomb.",
        "They built the tower as a monument to their greatness, reaching for the sky. Lightning split it on the night of their triumph. The proud were cast down from the heights in an instant. So fell their ambition, sudden and complete.",
    ],
    "wheel-01": [
        "Yesterday a king, today a beggar in the road, the old man watched the wheel of fortune turn. It had lifted him high and now cast him down. Such is fate, he thought, ever-turning, sparing no one. The wheel raises and ruins all in time.",
        "The gambler felt fortune's wheel spin in his favor at last. But the wheel never stops, and what it gives it takes again. By dawn his luck had turned and his winnings were gone. Fate is a wheel, fickle and endless.",
        "Round and round the great wheel turned, lifting the lowly and casting down the great. None could stop its turning or escape their place upon it. Kings became paupers and paupers kings. So the cycle of fortune ran, on and on.",
    ],
    "scales-01": [
        "The judge set down her gavel and the scales of justice tipped at last toward the truth. The guilty man's crimes were weighed against the law. Justice, blind and impartial, found him wanting. The scales had measured his soul and balanced the account.",
        "In the marble hall the statue of blind Justice held her scales aloft. Upon them the deeds of the dead were weighed, good against evil. The balance would decide each fate. So the scales rendered fair and final judgment.",
        "He stood before the scales of judgment, his good deeds in one pan and his sins in the other. The beam trembled, then settled. Justice would be done, exact and impartial. The scales weighed him and pronounced his sentence.",
    ],
    "salt-compass-01": [
        "Adrift and without bearings on the grey sea, he clutched the compass to his chest. Its needle swung north and gave him direction once more. The salt stung his lips as hope returned. He had found his way again.",
        "The old captain trusted the compass above all things; it had guided him through every storm. When all landmarks vanished, it pointed him true. Salt-crusted and worn, it never failed to show the way. By it he always found his bearings.",
    ],
    "key-01": [
        "She turned the ancient key in the rusted lock and the door swung open at last. Beyond it lay the secret she had sought all her life. The key had unlocked the truth and set her free. What was hidden was hidden no longer.",
        "The key had been lost for a hundred years, and with it the answer to the mystery. When the boy finally found it, the lock yielded. The door opened onto freedom and a new beginning. One small key had unlocked everything.",
        "He pressed the key into the lock of the forbidden room. It was the answer, the way in, the access he had been denied. With a click the secret was his. The key opened the door to all that had been closed.",
    ],
    # --- Italian arm (DRAFT — ONE model-authored source; native-review + augment) -----
    # Stale construals as short NARRATIVE SCENES in the generations' own register (same
    # rationale as the English narrative set). Seeds only — validate natively. Tracks
    # items_it.yaml (it-0.2.0-draft).
    "bilancia-01": [
        "Il giudice posò il martelletto e la bilancia della giustizia pendette infine verso la verità. Le colpe dell'imputato furono pesate contro la legge. La giustizia, cieca e imparziale, lo trovò mancante. La bilancia aveva misurato la sua anima.",
        "Nella sala di marmo la statua della Giustizia bendata reggeva alta la sua bilancia. Su di essa le opere dei morti venivano pesate, il bene contro il male. L'equilibrio avrebbe deciso ogni destino. Così la bilancia rendeva il giudizio finale.",
        "Egli stava davanti alla bilancia del giudizio, le buone azioni su un piatto e i peccati sull'altro. Il braccio tremò, poi si fermò. Giustizia sarebbe stata fatta, esatta e imparziale. La bilancia lo pesò e pronunciò la sentenza.",
    ],
    "specchietto-01": [
        "Davanti allo specchio la donna contava le prime rughe e piangeva la giovinezza svanita. Lo specchio non mentiva mai. Le restituiva la verità che nessuno osava dirle. In quel vetro vide tutto ciò che era stata e non era più.",
        "Il vecchio re guardò nello specchio e non riconobbe l'uomo crudele che lo fissava. Lo specchio gli mostrò l'anima nuda. Per la prima volta vide la verità di sé. Il vetro non concedeva né vanità né menzogna.",
    ],
    "campana-01": [
        "La campana suonò a morto sul paese e tutti chinarono il capo. Annunciava che un altro era passato. Il suo rintocco era la voce del lutto e della fine. Così la campana piangeva i morti come da secoli.",
        "All'alba la campana chiamò i fedeli alla messa, e il suo suono salì al cielo come una preghiera. I devoti accorsero alla casa di Dio. Era la voce della fede del villaggio. La campana univa tutti nella devozione.",
    ],
    "bambola-pane-01": [
        "Nella carestia la madre spezzò il pane e lo divise tra i figli affamati. Era la vita stessa che porgeva loro. Il pane era nutrimento, amore e sacrificio. Senza di esso non c'era domani.",
        "Il pane sull'altare era il corpo offerto, spezzato per tutti. Comunione e sacrificio in un solo gesto. I fedeli ne mangiarono in silenzio. Era la vita donata fatta cibo.",
    ],
    "gabbia-sasso-01": [
        "L'uccellino batteva le ali contro le sbarre della gabbia, sognando il cielo. La gabbia era la sua prigione. Ogni giorno cantava la libertà negata. Dentro quelle sbarre l'anima sognava di volare.",
        "Aprì infine la gabbia e l'uccello spiccò il volo verso il sole. La prigione era vuota. La libertà ritrovata riempì il cielo di canto. Nessuna sbarra avrebbe più trattenuto quel volo.",
    ],
    "ombrello-cucito-01": [
        "La pioggia cadeva fitta e lei aprì l'ombrello per ripararsi. Sotto quella cupola era al sicuro dalla tempesta. L'ombrello la proteggeva dal mondo che si rovesciava. Finché lo teneva alto, nulla poteva bagnarla.",
        "Il vecchio aprì l'ombrello contro il temporale della vita. Era il suo riparo, la sua difesa. Sotto di esso affrontava ogni avversità. L'ombrello lo proteggeva come una mano paterna.",
    ],
}

REFERENCE_SETS: dict[str, dict[str, list[str]]] = {
    "statement": CLICHE_REFERENCE,
    "narrative": CLICHE_REFERENCE_NARRATIVE,
}


class Embedder(Protocol):
    """Anything that turns texts into vectors. Injected so the pipeline runs under two
    different embedding families (§11.2 step 4). Implementations live in the runner."""

    name: str

    def embed(self, texts: list[str]) -> np.ndarray:  # (len(texts), dim)
        ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation so dot product == cosine similarity."""
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-12, None)


def nearest_cliche_distance(gen_vecs: np.ndarray, cliche_vecs: np.ndarray) -> np.ndarray:
    """Distance from the *nearest* stale construal: 1 − max cosine similarity to any
    cliché vector. Higher ⇒ farther from every obvious reading ⇒ more original.

    Nearest (not mean) because committing *one* cliché is enough to be unoriginal —
    we want the closest stale reading, not the average over a mixed reference set."""
    gv, cv = l2_normalize(gen_vecs), l2_normalize(cliche_vecs)
    sims = gv @ cv.T                       # (n_gen, n_cliche) cosine sims
    return 1.0 - sims.max(axis=1)          # (n_gen,) nearest-cliché distance


# ---------------------------------------------------------------------------
# Records & per-model aggregation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistanceRecord:
    """One generation's judge-free originality measurement."""
    model: str
    item: str
    generation_id: str
    distance: float            # nearest-cliché distance (higher = more original)
    judged_originality: float  # mean judge originality, for validation only (NaN if absent)


def per_model_mean(records: list[DistanceRecord], field: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in records:
        v = getattr(r, field)
        if not np.isnan(v):
            by.setdefault(r.model, []).append(v)
    return {m: float(np.mean(v)) for m, v in by.items() if v}


def distance_rows(records: list[DistanceRecord], embedder: str, reference: str,
                  extractor: str | None = None) -> list[dict]:
    """Flatten DistanceRecords into JSON-serialisable rows tagged with the run
    (embedder, reference, and optional construal `extractor`) — the persisted
    per-generation audit trail behind the §11.2 aggregates. `extractor` is None for
    whole-passage distance and the extractor's name for construal-extraction (§11.2 /
    `construal.py`). NaN judged-originality (no judge score) is written as null."""
    rows = []
    for r in records:
        jo = r.judged_originality
        rows.append({
            "embedder": embedder, "reference": reference, "extractor": extractor,
            "generation_id": r.generation_id, "model": r.model, "item": r.item,
            "distance": r.distance,
            "judged_originality": (None if np.isnan(jo) else jo),
        })
    return rows


# ---------------------------------------------------------------------------
# Step 2 — VALIDATION kill-switch. Run before any frontier test. The semantic metric
# must recover the tier structure, and in particular must NOT repeat the surface
# pilot's coherence-confound: the worst model (which scored spuriously HIGH on lexical
# novelty) must land near the cliché FLOOR (low distance), not high.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationResult:
    spearman: float            # per-model: distance vs judged originality (want strongly +)
    n_models: int
    worst_model: str
    worst_rank_by_distance: int  # 1 = highest distance; want it LOW-ranked (near floor)
    worst_in_bottom_half: bool
    passes: bool               # spearman ≥ threshold AND worst model near floor
    spearman_threshold: float = 0.5

    @property
    def fail_reason(self) -> str | None:
        """Distinguish the two failure modes — they have different implications.
        Coherence-confound (worst model ranks high) is the fatal surface-pilot failure;
        a merely weak Spearman with the canary at the floor is a *weak-signal* failure
        (the idea works in direction but lacks strength), not the confound."""
        if self.passes:
            return None
        if not self.worst_in_bottom_half:
            return "coherence-confound (worst model ranks HIGH on distance) — the surface-pilot failure; Route A closed"
        return (f"weak signal (Spearman {self.spearman:+.2f} < {self.spearman_threshold:g}, but canary at floor) "
                "— direction right, strength insufficient")

    def explain(self) -> str:
        verdict = "PASS — proceed to frontier test" if self.passes else f"FAIL — {self.fail_reason}"
        return (f"Spearman(distance, judged-orig) = {self.spearman:+.2f} over {self.n_models} models; "
                f"worst model {self.worst_model!r} ranks {self.worst_rank_by_distance}/{self.n_models} "
                f"by distance ({'near floor ✓' if self.worst_in_bottom_half else 'HIGH ✗ — coherence-confound'}). "
                f"{verdict}.")


def validate_against_tiers(records: list[DistanceRecord], worst_model: str,
                           spearman_threshold: float = 0.5) -> ValidationResult:
    dist = per_model_mean(records, "distance")
    orig = per_model_mean(records, "judged_originality")
    models = [m for m in dist if m in orig]
    rho = float(stats.spearmanr([dist[m] for m in models], [orig[m] for m in models]).statistic)
    ranked = sorted(dist, key=lambda m: dist[m], reverse=True)  # highest distance first
    rank = ranked.index(worst_model) + 1 if worst_model in ranked else len(ranked)
    bottom_half = rank > len(ranked) / 2
    return ValidationResult(
        spearman=rho, n_models=len(models), worst_model=worst_model,
        worst_rank_by_distance=rank, worst_in_bottom_half=bottom_half,
        passes=(rho >= spearman_threshold and bottom_half),
        spearman_threshold=spearman_threshold,
    )


# ---------------------------------------------------------------------------
# Step 3 — FRONTIER discrimination, with a flash positive control to make the null
# interpretable (§11.2). Paired by item, mirroring §4.9.
# ---------------------------------------------------------------------------
def _paired_by_item(records: list[DistanceRecord], a: str, b: str) -> tuple[float, float, int]:
    """Paired t-test on per-item mean distance, model a vs model b. Returns (meanΔ, p, n_items)."""
    da: dict[str, list[float]] = {}
    db: dict[str, list[float]] = {}
    for r in records:
        (da if r.model == a else db if r.model == b else {}).setdefault(r.item, []).append(r.distance)
    items = sorted(set(da) & set(db))
    if len(items) < 2:
        return (float("nan"), float("nan"), len(items))
    x = np.array([np.mean(da[i]) for i in items])
    y = np.array([np.mean(db[i]) for i in items])
    p = float(stats.ttest_rel(x, y).pvalue)
    return (float(x.mean() - y.mean()), p, len(items))


@dataclass(frozen=True)
class FrontierResult:
    trio_pairs: dict[tuple[str, str], tuple[float, float, int]]   # (a,b) -> (meanΔ, p, n)
    control_pairs: dict[tuple[str, str], tuple[float, float, int]]  # flagship vs flash
    alpha: float

    @property
    def control_resolves(self) -> bool:
        """The metric demonstrably resolves the KNOWN gap (flagship > flash)."""
        sig = [(d, p) for (d, p, n) in self.control_pairs.values() if not np.isnan(p)]
        return bool(sig) and all(p < self.alpha and d > 0 for d, p in sig)

    @property
    def trio_separates(self) -> bool:
        return any((not np.isnan(p)) and p < self.alpha for (_, p, _) in self.trio_pairs.values())

    def branch(self) -> str:
        if not self.control_resolves:
            return ("BRANCH B (metric ceiling): the metric cannot even resolve the known "
                    "flagship-vs-flash gap, so its within-trio result is UNINFORMATIVE about the frontier.")
        if self.trio_separates:
            return ("SEPARATION FOUND: the metric resolves the known gap AND separates the "
                    "flagship trio — a judge-free frontier ranking on cliché-distance.")
        return ("BRANCH A (real equivalence): the metric resolves flagship-vs-flash but NOT the "
                "trio — demonstrated resolution + genuine frontier equivalence on cliché-avoidance "
                "(same calibration logic as the flash result, §4.2).")


def frontier_separation(records: list[DistanceRecord], trio: list[str], control: str,
                        alpha: float = 0.05) -> FrontierResult:
    import itertools
    trio_pairs = {(a, b): _paired_by_item(records, a, b) for a, b in itertools.combinations(trio, 2)}
    control_pairs = {(m, control): _paired_by_item(records, m, control) for m in trio}
    return FrontierResult(trio_pairs=trio_pairs, control_pairs=control_pairs, alpha=alpha)


# ---------------------------------------------------------------------------
# Step 4 — embedder-as-prior empirical guard. Two embedding families must agree on the
# per-generation distance ranking, else the metric tracks embedder idiosyncrasy.
# ---------------------------------------------------------------------------
def two_embedder_agreement(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Spearman over per-generation distances under two embedders (keys = generation_id)."""
    keys = sorted(set(dist_a) & set(dist_b))
    if len(keys) < 3:
        return float("nan")
    return float(stats.spearmanr([dist_a[k] for k in keys], [dist_b[k] for k in keys]).statistic)


# ---------------------------------------------------------------------------
# STEP 5 — signal diagnostics (added to answer "is the weak rank signal real, and is
# the flash control just too close?"). All pure over records.
# ---------------------------------------------------------------------------
def generation_level_correlation(records: list[DistanceRecord]) -> dict | None:
    """The POWERED test for 'is there signal': correlate distance vs judged originality
    across all generations (n≈162), not just 9 model means. Model-level Spearman has no
    power at n=9; this does."""
    pairs = [(r.distance, r.judged_originality) for r in records if not np.isnan(r.judged_originality)]
    if len(pairs) < 3:
        return None
    x = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    pr = stats.pearsonr(x, y); sp = stats.spearmanr(x, y)
    return {"n": len(pairs), "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue)}


def model_level_spearman(records: list[DistanceRecord], n_perm: int = 10000, seed: int = 0) -> dict:
    """Model-mean distance vs judged originality, with a permutation p — so the +0.45-ish
    rank correlation gets a significance test instead of being eyeballed (n=9 is weak)."""
    dist = per_model_mean(records, "distance"); orig = per_model_mean(records, "judged_originality")
    models = [m for m in dist if m in orig]
    x = np.array([dist[m] for m in models]); y = np.array([orig[m] for m in models])
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    perm_p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return {"rho": rho, "perm_p": perm_p, "n_models": len(models)}


def normalize_within_item(records: list[DistanceRecord]) -> list[DistanceRecord]:
    """Z-score distance *within each item* before pooling — removes item-level variance
    (some objects sit closer to their clichés than others) that can swamp model
    differences, a cheap attempt to lift the signal-to-noise."""
    by_item: dict[str, list[float]] = {}
    for r in records:
        by_item.setdefault(r.item, []).append(r.distance)
    moments = {it: (float(np.mean(v)), float(np.std(v)) or 1.0) for it, v in by_item.items()}
    out = []
    for r in records:
        mu, sd = moments[r.item]
        out.append(DistanceRecord(model=r.model, item=r.item, generation_id=r.generation_id,
                                  distance=(r.distance - mu) / sd, judged_originality=r.judged_originality))
    return out


def group_contrast(records: list[DistanceRecord], group_a: list[str], group_b: list[str]) -> tuple[float, float, int]:
    """Paired-by-item mean-distance contrast between two model groups. Returns (meanΔ, p, n_items)."""
    A, B = set(group_a), set(group_b)
    a: dict[str, list[float]] = {}; b: dict[str, list[float]] = {}
    for r in records:
        if r.model in A: a.setdefault(r.item, []).append(r.distance)
        elif r.model in B: b.setdefault(r.item, []).append(r.distance)
    items = sorted(set(a) & set(b))
    if len(items) < 2:
        return (float("nan"), float("nan"), len(items))
    x = np.array([np.mean(a[i]) for i in items]); y = np.array([np.mean(b[i]) for i in items])
    return (float(x.mean() - y.mean()), float(stats.ttest_rel(x, y).pvalue), len(items))


def resolution_ladder(records: list[DistanceRecord], trio: list[str],
                      rungs: list[tuple[str, list[str]]]) -> list[dict]:
    """Trio-vs-opponent contrasts at increasing originality-gaps — replaces the single
    flash control with a calibrated ladder. Each rung: (label, opponent_models). Reports
    the judged-originality gap (so 'is flash too close?' is answered by comparison to
    bigger-gap rungs) alongside the metric's Δ and p. A metric with real resolution
    should light up the big-gap rungs even if it misses the small ones."""
    orig = per_model_mean(records, "judged_originality")
    trio_orig = float(np.mean([orig[m] for m in trio if m in orig])) if any(m in orig for m in trio) else float("nan")
    out = []
    for label, opponents in rungs:
        d, p, n = group_contrast(records, trio, opponents)
        opp_orig = [orig[m] for m in opponents if m in orig]
        gap = trio_orig - float(np.mean(opp_orig)) if opp_orig else float("nan")
        out.append({"rung": label, "judged_orig_gap": gap, "delta_distance": d, "p": p, "n_items": n,
                    "resolved": (not np.isnan(p)) and p < 0.05 and d > 0})
    return out
