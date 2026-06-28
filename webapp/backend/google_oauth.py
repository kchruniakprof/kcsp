"""Google OIDC login: Authorization Code flow via Authlib.

Two routes:
  GET /auth/google/login    -> redirect to Google consent screen
  GET /auth/google/callback -> exchange code, resolve user, set session cookie

State is persisted in the DB so OAuth callbacks work even when the client IP
changes between the login request and the callback (CGNAT / dual-WAN).
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from webapp.backend.auth import handle_google_user, make_session_token
from webapp.backend.db import OAuthState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/google")

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    client_kwargs={"scope": "openid email profile"},
)

# Authlib session key pattern: _state_{oauth_name}_{state_value}
_SESSION_KEY_PREFIX = "_state_google_"
_STATE_TTL_SECONDS = 600  # 10 minutes


def _set_session_cookie(response: RedirectResponse, token: str) -> None:
    is_prod = os.getenv("ENV", "dev") == "prod"
    response.set_cookie(
        "kcsp_session", token,
        httponly=True,
        samesite="lax",
        secure=is_prod,
        max_age=60 * 60 * 24 * 365,  # 1 year
        path="/kcsp",
    )


@router.get("/login")
async def login(request: Request):
    state = secrets.token_urlsafe(32)
    redirect_uri = str(request.url_for("google_callback"))

    # Authlib stores state + nonce in the session cookie
    response = await oauth.google.authorize_redirect(request, redirect_uri, state=state)

    # Also persist to DB so callbacks survive IP changes
    session_data = request.session.get(f"{_SESSION_KEY_PREFIX}{state}", {})
    nonce = session_data.get("data", {}).get("nonce")
    engine = request.app.state.engine
    with Session(engine) as db:
        # Evict expired states while we're here
        cutoff = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() - _STATE_TTL_SECONDS, tz=timezone.utc
        ).isoformat()
        expired = db.exec(select(OAuthState).where(OAuthState.created_at < cutoff)).all()
        for s in expired:
            db.delete(s)
        db.add(OAuthState(state=state, nonce=nonce,
                          created_at=datetime.now(timezone.utc).isoformat()))
        db.commit()

    return response


@router.get("/callback", name="google_callback")
async def callback(request: Request):
    state = request.query_params.get("state")

    if state:
        session_key = f"{_SESSION_KEY_PREFIX}{state}"
        if session_key not in request.session:
            # Client IP changed between login and callback — restore state from DB
            engine = request.app.state.engine
            with Session(engine) as db:
                record = db.get(OAuthState, state)
                if record:
                    redirect_uri = str(request.url_for("google_callback"))
                    request.session[session_key] = {
                        "data": {"redirect_uri": redirect_uri, "nonce": record.nonce}
                    }
                    logger.info("OAuth state restored from DB for client=%s state=%.8s…",
                                request.client, state)

        # Delete DB record regardless (one-time use)
        engine = request.app.state.engine
        with Session(engine) as db:
            record = db.get(OAuthState, state)
            if record:
                db.delete(record)
                db.commit()

    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        logger.warning("OAuth callback failed: %s | client=%s | state=%.8s…",
                       exc, request.client, state)
        return RedirectResponse(url="/kcsp/login?error=oauth_failed", status_code=302)

    claims = token.get("userinfo") or {}
    if not claims.get("email"):
        return RedirectResponse(url="/kcsp/login?error=no_email", status_code=302)

    engine = request.app.state.engine
    secret = request.app.state.session_secret
    admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()]

    with Session(engine) as session:
        user, redirect_path = handle_google_user(session, claims, admin_emails)
        session_token = make_session_token(user.id, secret)

    response = RedirectResponse(url=redirect_path, status_code=302)
    _set_session_cookie(response, session_token)
    return response
