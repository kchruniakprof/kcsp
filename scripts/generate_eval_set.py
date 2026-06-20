"""
Generate eval_set.yaml — 100 candidate questions grounded in real corpus sections.
LLM generates questions from actual Bedingungen text; expert reviews the YAML.

Distribution (PRD §7):
  COVERAGE_QUERY    40
  EXCLUSION_QUERY   20
  CLAIMS_PROCEDURE  15
  COMPARISON        10
  PRICE_QUOTE        5
  GENERAL_INFO       3
  COMPLAINT          2
  OUT_OF_SCOPE       5
  ─────────────── 100
"""
from __future__ import annotations

import json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import yaml
from groq import Groq
from src.hierarchy_parser import parse_all

CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
CORPUS = Path("D:/_FUN/kcsp/v1/sources/output_md")
OUT    = Path("D:/_FUN/kcsp/v1/eval_set.yaml")

# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------
DISTRIBUTION = {
    "COVERAGE_QUERY":   40,
    "EXCLUSION_QUERY":  20,
    "CLAIMS_PROCEDURE": 15,
    "COMPARISON":       10,
    "PRICE_QUOTE":       5,
    "GENERAL_INFO":      3,
    "COMPLAINT":         2,
    "OUT_OF_SCOPE":      5,
}

# Section type → preferred intent mapping
_TYPE_INTENT = {
    "WHAT_IS_INSURED":     "COVERAGE_QUERY",
    "EXCLUSIONS":          "EXCLUSION_QUERY",
    "CLAIMS_SETTLEMENT":   "CLAIMS_PROCEDURE",
    "LIMITS_COMPENSATION": "COVERAGE_QUERY",
    "OBLIGATIONS":         "CLAIMS_PROCEDURE",
    "PAYMENT":             "PRICE_QUOTE",
    "PRICING_DISCOUNT":    "PRICE_QUOTE",
    "TERM_CANCELLATION":   "GENERAL_INFO",
    "COMPLAINTS_LAW":      "COMPLAINT",
    "INSURED_PERSONS":     "COVERAGE_QUERY",
    "RISK_OBJECT":         "COVERAGE_QUERY",
    "WHERE_COVERED":       "COVERAGE_QUERY",
}

# Languages for cross-lingual variety
_LANG_POOL = ["de", "de", "de", "de", "pl", "en"]  # 4:1:1 ratio

_LANG_INSTRUCTION = {
    "de": "Stelle die Frage auf Deutsch.",
    "pl": "Zadaj pytanie po polsku (temat jest po niemiecku, ale pytanie po polsku).",
    "en": "Ask the question in English.",
}

# ---------------------------------------------------------------------------
# Generation system prompt
# ---------------------------------------------------------------------------
_SYS = """\
Du bist ein Qualitätssicherer für ein ERGO Versicherungs-RAG-System.
Deine Aufgabe: Generiere eine realistische Agentenfrage basierend auf dem gegebenen Versicherungsbedingungen-Abschnitt.

Die Frage soll:
- Von einem deutschen Versicherungsagenten oder Vertriebspartner kommen
- Konkret und präzise sein (kein Allgemeinwissen)
- Anhand des gegebenen Texts beantwortbar sein
- Dem angegebenen Intent entsprechen
- {lang_instruction}

Antworte NUR mit JSON:
{{
  "question": "<die Frage>",
  "expected_keywords": ["<schlüsselwort1>", "<schlüsselwort2>"],
  "expected_sparte": "<Kfz|Hausrat|Glas|Schmuck|null>",
  "expected_tarif": "<Spezial|Standard|Smart|Best|Best+Naturgefahren|Best+Fahrraddiebstahl|KT2021GLHR|KT Schmuck|null>",
  "notes": "<kurze Anmerkung für den Review>"
}}
"""

_OUT_OF_SCOPE_SYS = """\
Generiere eine Frage, die NICHT zu ERGO P&C-Sparten Kfz/Hausrat/Glas/Schmuck gehört.
Beispiele: Lebensversicherung, Reiseversicherung, Berufsunfähigkeit, allgemeine Steuerfragen.
Antworte NUR mit JSON:
{"question": "<frage>", "notes": "Out-of-scope test"}
"""

_COMPARISON_SYS = """\
Du bist ein Qualitätssicherer für ein ERGO Versicherungs-RAG-System.
Generiere eine Vergleichsfrage zwischen zwei Tarifen oder Sparten aus ERGO P&C.
Antworte NUR mit JSON:
{
  "question": "<vergleichsfrage>",
  "expected_keywords": ["<kw1>", "<kw2>"],
  "expected_sparte": "<Kfz|Hausrat|Glas|Schmuck|null>",
  "expected_tarif": null,
  "notes": "<anmerkung>"
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(system: str, user: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = CLIENT.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return {}


def _section_to_user(sec, intent: str, lang: str) -> str:
    lang_instr = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["de"])
    return (
        f"Sparte: {sec.sparte}  Tarif: {sec.tarif}\n"
        f"Abschnitt: {sec.heading}\n"
        f"Inhalt (Auszug):\n{sec.markdown[:1200]}\n\n"
        f"Intent: {intent}\n"
        f"Sprache der Frage: {lang_instr}"
    )


def _make_assert(kws: list[str], sparte: str | None, abstain_ok: bool = False) -> list[dict]:
    asserts = [{"type": "javascript", "value": "output.length > 0"}]
    if kws:
        asserts.append({
            "type": "icontains-any",
            "value": kws[:4],
        })
    if not abstain_ok:
        asserts.append({
            "type": "not-icontains",
            "value": "Traceback",
        })
    return asserts


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate():
    print("Parsing corpus...")
    docs = parse_all(CORPUS)
    sections = [s for doc in docs for s in doc.sections if s.section_code != "0"]
    print(f"  {len(sections)} sections from {len(docs)} docs\n")

    quota = dict(DISTRIBUTION)
    entries = []
    used_ids: set[int] = set()

    # --- OUT_OF_SCOPE: no corpus needed ---
    print("Generating OUT_OF_SCOPE...")
    for i in range(quota["OUT_OF_SCOPE"]):
        data = _call(_OUT_OF_SCOPE_SYS, f"Variante {i+1}")
        entries.append({
            "description": f"out_of_scope_{i+1:02d}",
            "vars": {
                "question": data.get("question", ""),
                "expected_intent": "OUT_OF_SCOPE",
                "expected_sparte": None,
                "notes": data.get("notes", ""),
            },
            "assert": [
                {"type": "javascript", "value": "output.length > 0"},
                {"type": "icontains-any", "values": ["Spezialist", "nicht", "keine", "außerhalb"]},
            ],
        })
        print(f"  [{i+1}/{quota['OUT_OF_SCOPE']}] {data.get('question','')[:60]}")
    quota["OUT_OF_SCOPE"] = 0

    # --- COMPARISON: cross-tarif ---
    comparison_prompts = [
        "Kfz Spezial vs Kfz Standard",
        "Hausrat Smart vs Hausrat Best",
        "Hausrat Best vs Hausrat Best+Naturgefahren",
        "Hausrat Best vs Hausrat Best+Fahrraddiebstahl",
        "Glasversicherung vs Hausrat",
        "Hausrat Smart vs Hausrat Best — Einbruchdiebstahl",
        "Kfz Spezial vs Kfz Standard — Kaskodeckung",
        "Schmuck vs Hausrat — Wertsachen",
        "Hausrat Best+Naturgefahren vs Hausrat Smart",
        "Glasversicherung vs Schmuckversicherung",
    ]
    print("\nGenerating COMPARISON...")
    for i in range(quota["COMPARISON"]):
        ctx = comparison_prompts[i % len(comparison_prompts)]
        data = _call(_COMPARISON_SYS, f"Vergleich: {ctx}")
        kws = data.get("expected_keywords", [])
        entries.append({
            "description": f"comparison_{i+1:02d}_{ctx.replace(' ','_')[:30]}",
            "vars": {
                "question": data.get("question", ""),
                "expected_intent": "COMPARISON",
                "expected_sparte": data.get("expected_sparte"),
                "notes": data.get("notes", ""),
            },
            "assert": _make_assert(kws, data.get("expected_sparte")),
        })
        print(f"  [{i+1}/{quota['COMPARISON']}] {data.get('question','')[:60]}")
    quota["COMPARISON"] = 0

    # --- Remaining intents from corpus sections ---
    remaining_intents = [(intent, n) for intent, n in quota.items() if n > 0]

    # Shuffle sections for variety
    random.seed(42)
    random.shuffle(sections)

    # Build intent→sections mapping by section_type
    intent_sections: dict[str, list] = {intent: [] for intent, _ in remaining_intents}
    for sec in sections:
        for stype in sec.section_types:
            mapped = _TYPE_INTENT.get(stype)
            if mapped in intent_sections:
                intent_sections[mapped].append(sec)

    # Also allow any section for GENERAL_INFO / COMPLAINT
    for intent in ["GENERAL_INFO", "COMPLAINT"]:
        if intent in intent_sections and len(intent_sections[intent]) < 5:
            intent_sections[intent].extend(sections[:20])

    total_corpus = sum(n for _, n in remaining_intents)
    generated = 0

    for intent, count in remaining_intents:
        pool = intent_sections.get(intent, sections)
        random.shuffle(pool)
        pool_iter = iter(pool * 10)  # repeat in case pool is small
        print(f"\nGenerating {intent} ({count} questions)...")

        for i in range(count):
            sec = next(pool_iter)
            lang = random.choice(_LANG_POOL)
            sys_prompt = _SYS.format(lang_instruction=_LANG_INSTRUCTION[lang])
            user_prompt = _section_to_user(sec, intent, lang)

            try:
                data = _call(sys_prompt, user_prompt)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            q = data.get("question", "")
            kws = data.get("expected_keywords", [])
            sp  = data.get("expected_sparte") or sec.sparte
            tf  = data.get("expected_tarif") or sec.tarif

            entries.append({
                "description": f"{intent.lower()}_{i+1:02d}_{sec.sparte}_{sec.tarif or 'none'}",
                "vars": {
                    "question": q,
                    "expected_intent": intent,
                    "expected_sparte": sp,
                    "expected_tarif": tf,
                    "source_section_id": sec.section_id,
                    "source_breadcrumb": sec.breadcrumb,
                    "language": lang,
                    "notes": data.get("notes", ""),
                },
                "assert": _make_assert(kws, sp),
            })
            generated += 1
            print(f"  [{i+1}/{count}] [{lang}] {q[:65]}")
            time.sleep(0.05)  # rate limit courtesy

    # ---------------------------------------------------------------------------
    # Write YAML
    # ---------------------------------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# ERGO P&C Agent-Bot — eval_set.yaml\n"
        "# GENERATED BY LLM — REVIEW REQUIRED\n"
        "# Instructions for reviewer:\n"
        "#   1. Check each 'question' is realistic for a German insurance agent\n"
        "#   2. Verify 'expected_keywords' are actually in the source conditions\n"
        "#   3. Adjust 'assert' blocks as needed\n"
        "#   4. Remove or flag questions that are incorrect/misleading\n"
        "#   5. Set REVIEWED: true in the header when done\n"
        "# REVIEWED: false\n\n"
    )

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(
            entries,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )

    total = len(entries)
    print(f"\n{'='*60}")
    print(f"Generated {total} questions -> {OUT}")
    print(f"{'='*60}\n")
    return total


if __name__ == "__main__":
    generate()
