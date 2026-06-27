"""FastAPI application factory for KCSP webapp."""
from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Engine
from sqlmodel import Session

from webapp.backend.db import create_db_and_tables, run_migrations, seed_settings


def create_app(engine: Optional[Engine] = None,
               session_secret: Optional[str] = None,
               rag=None,
               rag_factory: Optional[Callable] = None) -> FastAPI:
    from webapp.backend.db import get_engine

    _engine = engine or get_engine(os.getenv("DB_PATH", "/data/kcsp.db"))
    _secret = session_secret or os.getenv("SESSION_SECRET", "change-me-in-production-32chars!")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        create_db_and_tables(_engine)
        run_migrations(_engine)
        with Session(_engine) as session:
            seed_settings(session)
        app.state.engine = _engine
        app.state.session_secret = _secret
        app.state.rag_error = None
        # Use provided rag, custom factory, or auto-build from pipeline if env has API keys
        _rag = rag
        _factory = rag_factory or (
            (lambda: __import__("src.webapp_runner", fromlist=["KcspRunner"]).KcspRunner())
            if os.getenv("GROQ_API_KEY") else None
        )
        if _rag is None and _factory is not None:
            try:
                _rag = _factory()
                logger.info("RAG pipeline loaded OK")
            except Exception as exc:
                logger.error("RAG load failed: %s", exc, exc_info=True)
                app.state.rag_error = str(exc)
        app.state.rag = _rag
        yield

    app = FastAPI(title="KCSP Chatbot", root_path="/kcsp", lifespan=lifespan)

    # OAuth state/nonce store — separate cookie so it never clashes with the
    # JWT session cookie. Added before SessionMiddleware so proxy headers
    # (X-Forwarded-Proto/Host) are honored when building the redirect URI.
    from starlette.middleware.sessions import SessionMiddleware
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    app.add_middleware(SessionMiddleware, secret_key=_secret, session_cookie="oauth_state")
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

    from webapp.backend.chat import router as chat_router
    from webapp.backend.admin import router as admin_router
    app.include_router(chat_router)
    app.include_router(admin_router)

    @app.get("/debug-cookies")
    def debug_cookies(request: Request) -> dict:
        return {"cookies": dict(request.cookies), "headers": dict(request.headers)}

    @app.get("/health")
    def health(request: Request) -> dict:
        rag_error = getattr(request.app.state, "rag_error", None)
        rag_loaded = getattr(request.app.state, "rag", None) is not None
        if rag_error:
            rag_status = "failed"
        elif rag_loaded:
            rag_status = "ok"
        else:
            rag_status = "unavailable"
        result: dict = {"status": "ok", "rag": rag_status}
        if rag_error:
            result["rag_error"] = rag_error
        return result

    @app.get("/me")
    def me(request: Request):
        from webapp.backend.auth import get_current_user
        token = request.cookies.get("kcsp_session")
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")
        with Session(_engine) as session:
            user = get_current_user(session, token, _secret)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return {"email": user.email, "name": user.name,
                "role": user.role, "status": user.status,
                "budget_eur": user.budget_eur, "spent_eur": user.spent_eur}

    from webapp.backend.auth_routes import router as auth_router
    from webapp.backend.google_oauth import router as google_router
    app.include_router(auth_router)
    app.include_router(google_router)

    # static assets (JS/CSS/etc.)
    # Note: app.mount("/assets", StaticFiles(...)) breaks with root_path set in Starlette 1.x
    # because get_route_path() strips the wrong prefix. Use explicit routes instead.
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        assets_dir = static_dir / "assets"

        if assets_dir.exists():
            _assets_dir = assets_dir

            @app.get("/assets/{asset_path:path}")
            async def serve_asset(asset_path: str):
                file_path = _assets_dir / asset_path
                try:
                    file_path.resolve().relative_to(_assets_dir.resolve())
                except ValueError:
                    raise HTTPException(status_code=404)
                if not file_path.is_file():
                    raise HTTPException(status_code=404)
                return FileResponse(str(file_path))

        # SPA catch-all — serve index.html for all unmatched routes
        index_html = static_dir / "index.html"

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            if index_html.exists():
                return FileResponse(str(index_html))
            raise HTTPException(status_code=404)

    return app


# Module-level app for uvicorn (production / dev server)
app = create_app()
