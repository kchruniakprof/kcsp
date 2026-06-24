"""Admin endpoints — user management, metrics, settings."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session, func, select

from webapp.backend.db import Message, Settings, Thread, User
from webapp.backend.chat import get_db_session, get_current_active_user, _build_trace_payload

router = APIRouter(prefix="/admin")


def require_admin(user: User = Depends(get_current_active_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# ---------- request models ----------

class StatusAction(BaseModel):
    action: str  # approve|reject|block|unblock


class BudgetUpdate(BaseModel):
    budget_eur: float


class SettingsUpdate(BaseModel):
    default_budget_eur: Optional[float] = None
    global_daily_limit_eur: Optional[float] = None
    kill_switch: Optional[bool] = None
    usd_eur_rate: Optional[float] = None


# ---------- endpoints ----------

@router.get("/users")
def list_users(admin: User = Depends(require_admin),
               session: Session = Depends(get_db_session)):
    users = session.exec(select(User)).all()
    return [{"id": u.id, "email": u.email, "name": u.name, "role": u.role,
             "status": u.status, "budget_eur": u.budget_eur, "spent_eur": u.spent_eur,
             "last_active_at": u.last_active_at} for u in users]


@router.post("/users/{user_id}/status")
def update_user_status(user_id: int, body: StatusAction,
                       admin: User = Depends(require_admin),
                       session: Session = Depends(get_db_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404)
    action_map = {
        "approve": "active",
        "reject":  "blocked",
        "block":   "blocked",
        "unblock": "active",
    }
    new_status = action_map.get(body.action)
    if not new_status:
        raise HTTPException(status_code=422, detail=f"Unknown action: {body.action}")
    user.status = new_status
    session.add(user)
    session.commit()
    return {"id": user_id, "status": user.status}


@router.post("/users/{user_id}/budget")
def update_user_budget(user_id: int, body: BudgetUpdate,
                       admin: User = Depends(require_admin),
                       session: Session = Depends(get_db_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404)
    user.budget_eur = body.budget_eur
    session.add(user)
    session.commit()
    return {"id": user_id, "budget_eur": user.budget_eur}


@router.get("/users/{user_id}/history")
def user_history(user_id: int, offset: int = 0,
                 admin: User = Depends(require_admin),
                 session: Session = Depends(get_db_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404)

    # Fetch assistant messages DESC with limit/offset
    asst_msgs = session.exec(
        select(Message)
        .where(Message.user_id == user_id, Message.role == "assistant")
        .order_by(Message.id.desc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(100)
    ).all()

    # Fetch threads for titles
    thread_ids = list({m.thread_id for m in asst_msgs})
    threads = {t.id: t for t in session.exec(
        select(Thread).where(Thread.id.in_(thread_ids))  # type: ignore[attr-defined]
    ).all()} if thread_ids else {}

    # Pair each assistant-msg with the preceding user-msg in the same thread
    result = []
    for asst in asst_msgs:
        user_msg = session.exec(
            select(Message)
            .where(
                Message.thread_id == asst.thread_id,
                Message.role == "user",
                Message.id < asst.id,
            )
            .order_by(Message.id.desc())  # type: ignore[attr-defined]
            .limit(1)
        ).first()
        result.append({
            "assistant_message_id": asst.id,
            "question_text": user_msg.content_markdown if user_msg else None,
            "answer_text": asst.content_markdown,
            "answer_status": asst.status,
            "abstained": asst.abstained,
            "cached": asst.cached,
            "cost_eur": asst.cost_eur,
            "created_at": asst.created_at,
            "thread_title": threads[asst.thread_id].title if asst.thread_id in threads else None,
        })
    return result


@router.get("/metrics")
def metrics(admin: User = Depends(require_admin),
            session: Session = Depends(get_db_session)):
    settings = session.get(Settings, 1)
    cost_today = settings.spent_today_eur if settings else 0.0

    total_q = session.exec(
        select(func.count(Message.id)).where(Message.role == "assistant")
    ).one()

    abstained = session.exec(
        select(func.count(Message.id)).where(
            Message.role == "assistant", Message.abstained == True  # noqa: E712
        )
    ).one()

    active_count = session.exec(
        select(func.count(User.id)).where(User.status == "active")
    ).one()

    pending_count = session.exec(
        select(func.count(User.id)).where(User.status == "pending")
    ).one()

    cost_month = session.exec(
        select(func.sum(Message.cost_eur)).where(Message.role == "assistant")
    ).one() or 0.0

    abstain_rate = (abstained / total_q) if total_q else 0.0

    return {
        "cost_today": cost_today,
        "cost_month": cost_month,
        "total_questions": total_q,
        "abstain_rate": round(abstain_rate, 4),
        "active_count": active_count,
        "pending_count": pending_count,
        "kill_switch": settings.kill_switch if settings else False,
    }


@router.get("/messages/{message_id}/trace")
def admin_get_trace(message_id: int,
                    admin: User = Depends(require_admin),
                    session: Session = Depends(get_db_session)):
    msg = session.get(Message, message_id)
    if not msg:
        raise HTTPException(status_code=404)
    return _build_trace_payload(message_id, session)


@router.post("/settings")
def update_settings(body: SettingsUpdate,
                    admin: User = Depends(require_admin),
                    session: Session = Depends(get_db_session)):
    settings = session.get(Settings, 1)
    if not settings:
        raise HTTPException(status_code=500, detail="Settings not seeded")
    if body.default_budget_eur is not None:
        settings.default_budget_eur = body.default_budget_eur
    if body.global_daily_limit_eur is not None:
        settings.global_daily_limit_eur = body.global_daily_limit_eur
    if body.kill_switch is not None:
        settings.kill_switch = body.kill_switch
    if body.usd_eur_rate is not None:
        settings.usd_eur_rate = body.usd_eur_rate
    session.add(settings)
    session.commit()
    return {"ok": True}
