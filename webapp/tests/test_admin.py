"""Tests for admin endpoints."""
from fastapi.testclient import TestClient
from webapp.backend.main import create_app
from webapp.backend.db import get_engine, User, Session
from webapp.backend.auth import make_session_token

def make_admin_client():
    engine = get_engine(":memory:")
    app = create_app(engine=engine, session_secret="test-secret-32chars-padding!!")
    client = TestClient(app, raise_server_exceptions=False)
    # Trigger lifespan to create tables
    client.__enter__()
    with Session(engine) as s:
        admin = User(email="admin@test.com", name="Admin", role="admin", status="active")
        s.add(admin); s.commit(); s.refresh(admin)
        token = make_session_token(admin.id, "test-secret-32chars-padding!!")
    client.cookies.set("kcsp_session", token)
    return client, engine

def test_list_users():
    client, _ = make_admin_client()
    r = client.get("/admin/users")
    assert r.status_code == 200

def test_approve_user():
    client, engine = make_admin_client()
    with Session(engine) as s:
        u = User(email="pending@test.com", name="Pending", role="user", status="pending")
        s.add(u); s.commit(); s.refresh(u)
        uid = u.id
    r = client.post(f"/admin/users/{uid}/status", json={"action": "approve"})
    assert r.status_code == 200
    assert r.json()["status"] == "active"

def test_block_user():
    client, engine = make_admin_client()
    with Session(engine) as s:
        u = User(email="active@test.com", name="Active", role="user", status="active")
        s.add(u); s.commit(); s.refresh(u)
        uid = u.id
    r = client.post(f"/admin/users/{uid}/status", json={"action": "block"})
    assert r.status_code == 200
    assert r.json()["status"] == "blocked"

def test_kill_switch():
    client, _ = make_admin_client()
    r = client.post("/admin/settings", json={"kill_switch": True})
    assert r.status_code == 200
    r2 = client.get("/admin/metrics")
    assert r2.json()["kill_switch"] is True
