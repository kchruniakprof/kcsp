"""
Hybrid product detector: Aho-Corasick → RapidFuzz → BGE-M3 → LLM fallback.
Layers 1+2 are deterministic; 3+4 require embeddings / Groq (built lazily).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import ahocorasick
from rapidfuzz import process, fuzz

from src.observability import get_logger

_log = get_logger("product_detector")

# ---------------------------------------------------------------------------
# Catalog: all searchable product terms → (sparte, tarif)
# ---------------------------------------------------------------------------

_CATALOG: list[tuple[str, str, Optional[str]]] = [
    # (keyword, sparte, tarif_or_None)
    # Kfz
    ("Kfz-Haftpflicht",     "Kfz",     None),
    ("Kfz Haftpflicht",     "Kfz",     None),
    ("KFZ",                 "Kfz",     None),
    ("Kfz",                 "Kfz",     None),
    ("Kraftfahrzeug",       "Kfz",     None),
    ("AKB",                 "Kfz",     None),
    ("Kfz Spezial",         "Kfz",     "Spezial"),
    ("Kfz Standard",        "Kfz",     "Standard"),
    ("Kfz-Versicherung Spezial", "Kfz", "Spezial"),
    ("AKB Spezial",         "Kfz",     "Spezial"),
    # Hausrat — tarife (longer/more specific first for AC priority)
    ("Hausrat Best Naturgefahren",      "Hausrat", "Best+Naturgefahren"),
    ("Hausrat Best Fahrraddiebstahl",   "Hausrat", "Best+Fahrraddiebstahl"),
    ("Best Naturgefahren",              "Hausrat", "Best+Naturgefahren"),
    ("Best Fahrraddiebstahl",           "Hausrat", "Best+Fahrraddiebstahl"),
    ("Hausrat Best",        "Hausrat",  "Best"),
    ("Hausrat Smart",       "Hausrat",  "Smart"),
    ("Hausrat",             "Hausrat",  None),
    ("Hausratversicherung", "Hausrat",  None),
    # Glas
    ("Glasversicherung",    "Glas",     "KT2021GLHR"),
    ("Glasbruch",           "Glas",     None),
    ("Verglasung",          "Glas",     None),
    ("Glas",                "Glas",     None),
    # Schmuck
    ("Schmuckversicherung", "Schmuck",  "KT Schmuck"),
    ("Schmuck",             "Schmuck",  None),
    ("Pelzsachen",          "Schmuck",  None),
    ("Wertsachen",          "Schmuck",  None),
]

# Fuzzy candidates (term → sparte, tarif) for RapidFuzz
# Uses fuzz.ratio (whole-string), so longer compound terms beat suffix collisions
_FUZZY_TERMS: list[tuple[str, str, Optional[str]]] = [
    ("Kfz-Haftpflicht", "Kfz",     None),
    ("Kfz",             "Kfz",     None),
    ("Kraftfahrzeug",   "Kfz",     None),
    ("Hausrat",         "Hausrat", None),
    ("Hausrat Smart",   "Hausrat", "Smart"),
    ("Hausrat Best",    "Hausrat", "Best"),
    ("Glasversicherung","Glas",    "KT2021GLHR"),
    ("Schmuck",         "Schmuck", None),
]

_FUZZY_KEYS = [t[0] for t in _FUZZY_TERMS]


@dataclass
class DetectionResult:
    sparte: Optional[str] = None
    tarif: Optional[str] = None
    confidence: float = 0.0
    layer_used: int = 0


class HybridProductDetector:
    def __init__(self) -> None:
        self._ac = self._build_ac()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ac() -> ahocorasick.Automaton:
        A = ahocorasick.Automaton()
        for keyword, sparte, tarif in _CATALOG:
            A.add_word(keyword.lower(), (keyword, sparte, tarif))
            # also add without hyphen variant
            alt = keyword.lower().replace("-", " ")
            if alt != keyword.lower():
                A.add_word(alt, (keyword, sparte, tarif))
        A.make_automaton()
        return A

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, query: str) -> DetectionResult:
        _log.info("step_start", step="product_detector", query=query)

        result = self._layer1_ac(query)
        if result.confidence >= 0.95:
            _log.info("step_done", step="product_detector", layer=1,
                      sparte=result.sparte, tarif=result.tarif, confidence=result.confidence)
            return result

        result = self._layer2_fuzzy(query)
        if result.confidence >= 0.7:
            _log.info("step_done", step="product_detector", layer=2,
                      sparte=result.sparte, tarif=result.tarif, confidence=result.confidence)
            return result

        # Layers 3+4 (embeddings / LLM) — not yet implemented; return low-conf
        _log.info("step_done", step="product_detector", layer=0,
                  sparte=None, tarif=None, confidence=0.0)
        return DetectionResult(sparte=None, tarif=None, confidence=0.0, layer_used=0)

    # ------------------------------------------------------------------
    # Layer 1: Aho-Corasick exact match
    # ------------------------------------------------------------------

    def _layer1_ac(self, query: str) -> DetectionResult:
        q = query.lower()
        best: Optional[tuple[str, str, Optional[str]]] = None
        best_len = 0

        for _end, (kw, sparte, tarif) in self._ac.iter(q):
            if len(kw) > best_len:
                best_len = len(kw)
                best = (kw, sparte, tarif)

        if best:
            _, sparte, tarif = best
            return DetectionResult(sparte=sparte, tarif=tarif, confidence=0.99, layer_used=1)

        return DetectionResult(confidence=0.0, layer_used=0)

    # ------------------------------------------------------------------
    # Layer 2: RapidFuzz typo-tolerant
    # ------------------------------------------------------------------

    def _layer2_fuzzy(self, query: str) -> DetectionResult:
        tokens = query.split()
        # candidate substrings: full query + 1-word + 2-word windows
        candidates = [query] + tokens + [
            f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)
        ]

        best_score = 0
        best_idx = -1
        for candidate in candidates:
            hit = process.extractOne(
                candidate,
                _FUZZY_KEYS,
                scorer=fuzz.ratio,
                score_cutoff=80,
            )
            if hit and hit[1] > best_score:
                best_score = hit[1]
                best_idx = hit[2]

        if best_idx >= 0:
            _, sparte, tarif = _FUZZY_TERMS[best_idx]
            return DetectionResult(sparte=sparte, tarif=tarif,
                                   confidence=best_score / 100.0, layer_used=2)

        return DetectionResult(confidence=0.0, layer_used=0)
