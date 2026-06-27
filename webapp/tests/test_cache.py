"""Tests for cache utilities."""
from webapp.backend.cache import norm_query, make_query_hash, build_version_tag

def test_norm_query():
    assert norm_query("  Hello  World  ") == "hello world"

def test_make_query_hash_stable():
    h1 = make_query_hash("test", "v1")
    h2 = make_query_hash("test", "v1")
    assert h1 == h2

def test_make_query_hash_version_sensitive():
    h1 = make_query_hash("test", "v1")
    h2 = make_query_hash("test", "v2")
    assert h1 != h2
