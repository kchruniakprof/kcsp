"""Tests for threads CRUD + ask endpoint with stub RAG."""
from fastapi.testclient import TestClient
from webapp.backend.main import create_app
from webapp.backend.db import get_engine

class StubRag:
    def run(self, query, stage_callback=None):
        from types import SimpleNamespace
        if stage_callback:
            for stage in ["query_expansion", "retrieval", "generation", "critic"]:
                stage_callback(stage)
        return SimpleNamespace(
            answer_markdown="Test answer",
            abstained=False,
            cited_sources=[1, 2],
            retrieved_doc_ids=[1, 2],
            used_brute_force=False,
            audit_metadata={},
        )

def make_client_with_active_user():
    from webapp.backend.db import User, Session
    from webapp.backend.auth import make_session_token
    engine = get_engine(":memory:")
    app = create_app(engine=engine, session_secret="test-secret-32chars-padding!!", rag=StubRag())
    # Use context manager to trigger lifespan (creates tables)
    client = TestClient(app, raise_server_exceptions=False)
    # Trigger lifespan by entering client context
    client.__enter__()
    # Create active user after tables exist
    with Session(engine) as s:
        u = User(email="user@test.com", name="Test User", role="user", status="active")
        s.add(u); s.commit(); s.refresh(u)
        token = make_session_token(u.id, "test-secret-32chars-padding!!")
    client.cookies.set("kcsp_session", token)
    return client

def test_create_and_list_threads():
    client = make_client_with_active_user()
    r = client.post("/threads", json={"title": "My thread"})
    assert r.status_code == 200
    tid = r.json()["id"]
    r2 = client.get("/threads")
    assert any(t["id"] == tid for t in r2.json())

def test_delete_thread():
    client = make_client_with_active_user()
    r = client.post("/threads", json={"title": "Del me"})
    tid = r.json()["id"]
    r2 = client.delete(f"/threads/{tid}")
    assert r2.status_code == 204

def test_rename_thread():
    client = make_client_with_active_user()
    r = client.post("/threads", json={"title": "Old"})
    tid = r.json()["id"]
    r2 = client.patch(f"/threads/{tid}", json={"title": "New"})
    assert r2.json()["title"] == "New"
