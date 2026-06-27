"""Integration tests for auth flow using in-memory SQLite + TestClient."""
from fastapi.testclient import TestClient
from webapp.backend.main import create_app
from webapp.backend.db import get_engine

def make_client(admin_emails="admin@test.com"):
    engine = get_engine(":memory:")
    app = create_app(engine=engine, session_secret="test-secret-32chars-padding!!")
    # override ADMIN_EMAILS env for test
    import os; os.environ["ADMIN_EMAILS"] = admin_emails
    return TestClient(app, raise_server_exceptions=True)

def test_health():
    client = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_me_unauthenticated():
    client = make_client()
    r = client.get("/me")
    assert r.status_code == 401

def test_logout():
    client = make_client()
    r = client.post("/auth/logout")
    assert r.status_code == 204
