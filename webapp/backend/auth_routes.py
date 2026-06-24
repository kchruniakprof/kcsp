"""Session lifecycle routes. Login is handled by Google OIDC (google_oauth.py)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/auth")


@router.post("/logout", status_code=204)
async def logout():
    response = JSONResponse(None, status_code=204)
    response.delete_cookie("session", path="/kcsp")
    return response
