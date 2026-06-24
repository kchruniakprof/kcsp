"""Google OIDC login: Authorization Code flow via Authlib.

Two routes:
  GET /auth/google/login    -> redirect to Google consent screen
  GET /auth/google/callback -> exchange code, resolve user, set session cookie

The callback is a thin wrapper around handle_google_user (pure domain logic).
"""
from __future__ import annotations

import os

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session

from webapp.backend.auth import handle_google_user, make_session_token

router = APIRouter(prefix="/auth/google")

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    client_kwargs={"scope": "openid email profile"},
)


def _set_session_cookie(response: RedirectResponse, token: str) -> None:
    is_prod = os.getenv("ENV", "dev") == "prod"
    response.set_cookie(
        "session", token,
        httponly=True,
        samesite="lax",
        secure=is_prod,
        max_age=60 * 60 * 24 * 365,  # 1 year
        path="/kcsp",
    )


@router.get("/login")
async def login(request: Request):
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="google_callback")
async def callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
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
