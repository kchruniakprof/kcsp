"""Budget enforcement and cost deduction."""
from __future__ import annotations

from fastapi import HTTPException
from sqlmodel import Session

from webapp.backend.db import Settings, User


def check_budget(session: Session, user: User) -> None:
    settings = session.get(Settings, 1)

    if settings and settings.kill_switch:
        raise HTTPException(status_code=503, detail="Bot paused by admin")

    if settings and settings.spent_today_eur >= settings.global_daily_limit_eur:
        raise HTTPException(status_code=503, detail="Global daily limit reached")

    if user.spent_eur >= user.budget_eur:
        raise HTTPException(status_code=402, detail="Token budget exhausted")


def deduct_cost(session: Session, user: User, cost_eur: float) -> None:
    user.spent_eur = (user.spent_eur or 0.0) + cost_eur
    session.add(user)

    settings = session.get(Settings, 1)
    if settings:
        settings.spent_today_eur = (settings.spent_today_eur or 0.0) + cost_eur
        session.add(settings)

    session.commit()
