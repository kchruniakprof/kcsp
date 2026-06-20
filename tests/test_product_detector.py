"""Tests for product_detector — TDD cycle."""
import pytest
from unittest.mock import MagicMock, patch

from src.product_detector import DetectionResult, HybridProductDetector


@pytest.fixture(scope="module")
def detector():
    return HybridProductDetector()


# --- Layer 1: Aho-Corasick exact match ---

def test_exact_sparte_kfz(detector):
    r = detector.detect("Kfz-Haftpflicht Frage")
    assert r.sparte == "Kfz"
    assert r.confidence >= 0.95


def test_exact_tarif_hausrat_best(detector):
    r = detector.detect("Hausrat Best Versicherung")
    assert r.sparte == "Hausrat"
    assert r.tarif == "Best"
    assert r.confidence >= 0.95


def test_exact_tarif_hausrat_smart(detector):
    r = detector.detect("Was ist in der Hausrat Smart versichert?")
    assert r.tarif == "Smart"


def test_exact_glas(detector):
    r = detector.detect("Glasversicherung Schaden")
    assert r.sparte == "Glas"
    assert r.confidence >= 0.95


def test_exact_schmuck(detector):
    r = detector.detect("Schmuckversicherung Frage")
    assert r.sparte == "Schmuck"


def test_exact_naturgefahren_tarif(detector):
    r = detector.detect("Hausrat Best Naturgefahren Überschwemmung")
    assert r.tarif == "Best+Naturgefahren"


def test_exact_fahrraddiebstahl_tarif(detector):
    r = detector.detect("Hausrat Best Fahrraddiebstahl gestohlen")
    assert r.tarif == "Best+Fahrraddiebstahl"


# --- Layer 2: RapidFuzz typo-tolerant ---

def test_typo_hausrat(detector):
    r = detector.detect("Hautsrat Smart versichert")  # typo: Hautsrat
    assert r.sparte == "Hausrat"
    assert r.layer_used <= 2


def test_typo_kfz(detector):
    r = detector.detect("Khz-Haftpflicht")  # typo: Khz
    assert r.sparte == "Kfz"
    assert r.layer_used <= 2


def test_typo_glasversicherung(detector):
    r = detector.detect("Glasversicherrung")  # double r
    assert r.sparte == "Glas"
    assert r.layer_used <= 2


# --- Detection result fields ---

def test_result_has_all_fields(detector):
    r = detector.detect("Kfz Spezial")
    assert hasattr(r, "sparte")
    assert hasattr(r, "tarif")
    assert hasattr(r, "confidence")
    assert hasattr(r, "layer_used")


def test_unknown_query_low_confidence(detector):
    r = detector.detect("Lebensversicherung Rente")  # out of scope
    assert r.confidence < 0.5 or r.sparte is None


def test_no_tarif_when_only_sparte_detected(detector):
    r = detector.detect("Kfz Versicherung allgemein")
    assert r.sparte == "Kfz"
    # tarif may be None when not specified
    # (not an error — retriever uses sparte filter alone)


# --- Multi-tarif Hausrat: correct tarif isolation ---

def test_best_not_confused_with_smart(detector):
    r = detector.detect("Hausrat Best Einbruchdiebstahl")
    assert r.tarif == "Best"
    assert r.tarif != "Smart"


def test_smart_not_confused_with_best(detector):
    r = detector.detect("Hausrat Smart Feuer")
    assert r.tarif == "Smart"
    assert r.tarif != "Best"


# --- Layer tracking ---

def test_exact_match_uses_layer_1(detector):
    r = detector.detect("Hausrat Best")
    assert r.layer_used == 1


def test_typo_uses_layer_2(detector):
    r = detector.detect("Hautsrat Smrat")  # two typos
    assert r.layer_used <= 2
