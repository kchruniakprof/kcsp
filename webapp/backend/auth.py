"""Auth helpers: session tokens + user lifecycle (Google login only)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import jwt
from sqlmodel import Session, select

from webapp.backend.db import User


_ALGORITHM = "HS256"


def make_session_token(user_id: int, secret: str) -> str:
    payload = {"sub": str(user_id), "iat": datetime.now(timezone.utc).timestamp()}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def get_current_user(session: Session, token: Optional[str], secret: str) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=[_ALGORITHM])
        user_id = int(payload["sub"])
    except Exception:
        return None
    return session.get(User, user_id)


_STATUS_REDIRECT = {"active": "/kcsp/chat", "pending": "/kcsp/pending", "blocked": "/kcsp/blocked"}


def handle_google_user(
    session: Session, claims: dict, admin_emails: list[str]
) -> tuple[User, str]:
    """Map verified Google claims to a user + the path to redirect them to.

    Upserts by email (admin bootstrap handled by upsert_user_by_email), stores
    the Google display name, and picks a redirect path from the account status.
    Knows nothing about HTTP or the OAuth client.
    """
    email = claims["email"].strip().lower()
    user = upsert_user_by_email(session, email, admin_emails)

    name = (claims.get("name") or "").strip()
    if name and user.name != name:
        user.name = name
        session.add(user)
        session.commit()
        session.refresh(user)

    redirect_path = _STATUS_REDIRECT.get(user.status, "/kcsp/pending")
    return user, redirect_path


def upsert_user_by_email(session: Session, email: str, admin_emails: list[str]) -> User:
    is_admin = email in admin_emails
    user = session.exec(select(User).where(User.email == email)).first()
    now = datetime.now(timezone.utc).isoformat()

    if user is None:
        user = User(
            email=email,
            role="admin" if is_admin else "user",
            status="active" if is_admin else "pending",
            created_at=now,
            last_active_at=now,
        )
    else:
        user.last_active_at = now
        if is_admin:
            user.role = "admin"
            user.status = "active"

    session.add(user)
    session.commit()
    session.refresh(user)
    return user
