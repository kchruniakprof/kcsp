"""
Domain constants for KCSP — single source of truth.

SPARTES: P&C insurance branches supported by ERGO pipeline.
SECTION_TYPES: section types used for retrieval routing (QueryExpansion → DocFilter).
SECTION_TYPES_FULL: all 16 section types including enrichment-only types
    (INSURER_ID, PRODUCT_STRUCTURE, CONTRACT_FORMATION, SPECIAL_PROVISIONS)
    that are not exposed in QueryExpansion because they don't aid retrieval routing.
"""

SPARTES: frozenset[str] = frozenset({"Kfz", "Hausrat", "Glas", "Schmuck"})

SECTION_TYPES: list[str] = [
    "WHAT_IS_INSURED",
    "EXCLUSIONS",
    "CLAIMS_SETTLEMENT",
    "LIMITS_COMPENSATION",
    "OBLIGATIONS",
    "PAYMENT",
    "PRICING_DISCOUNT",
    "TERM_CANCELLATION",
    "COMPLAINTS_LAW",
    "INSURED_PERSONS",
    "RISK_OBJECT",
    "WHERE_COVERED",
]

SECTION_TYPES_FULL: list[str] = SECTION_TYPES + [
    "INSURER_ID",
    "PRODUCT_STRUCTURE",
    "CONTRACT_FORMATION",
    "SPECIAL_PROVISIONS",
]
