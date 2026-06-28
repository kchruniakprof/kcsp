"""Chat endpoints: threads, ask, status polling."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from webapp.backend.budget import check_budget, deduct_cost
from webapp.backend.db import AnswerCache, Message, Thread, Trace, User
from sqlmodel import delete as sql_delete

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4)
_semaphore: Optional[asyncio.Semaphore] = None


def get_semaphore(max_concurrency: int = 3) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max_concurrency)
    return _semaphore


# ---------- request/response models ----------

class ThreadCreate(BaseModel):
    title: str = ""


class AskRequest(BaseModel):
    query: str


class ThreadRename(BaseModel):
    title: str


# ---------- deps ----------

def _get_session_and_secret(request: Request):
    return request.app.state.engine, request.app.state.session_secret


def get_db_session(request: Request):
    engine = request.app.state.engine
    with Session(engine) as session:
        yield session


def get_current_active_user(request: Request, session: Session = Depends(get_db_session)):
    from webapp.backend.auth import get_current_user
    secret = request.app.state.session_secret
    token = request.cookies.get("kcsp_session")
    user = get_current_user(session, token, secret)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.status == "blocked":
        raise HTTPException(status_code=403, detail="Access blocked")
    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Account pending approval")
    return user


# ---------- thread endpoints ----------

@router.post("/threads")
def create_thread(body: ThreadCreate, request: Request,
                  user: User = Depends(get_current_active_user),
                  session: Session = Depends(get_db_session)):
    now = datetime.now(timezone.utc).isoformat()
    thread = Thread(user_id=user.id, title=body.title or "New thread",
                    created_at=now, updated_at=now)
    session.add(thread)
    session.commit()
    session.refresh(thread)
    return {"id": thread.id, "title": thread.title}


@router.get("/threads")
def list_threads(request: Request,
                 user: User = Depends(get_current_active_user),
                 session: Session = Depends(get_db_session)):
    threads = session.exec(
        select(Thread).where(Thread.user_id == user.id)
    ).all()
    return [{"id": t.id, "title": t.title, "updated_at": t.updated_at} for t in threads]


@router.get("/threads/{thread_id}/messages")
def list_messages(thread_id: int, request: Request,
                  user: User = Depends(get_current_active_user),
                  session: Session = Depends(get_db_session)):
    thread = session.get(Thread, thread_id)
    if not thread or thread.user_id != user.id:
        raise HTTPException(status_code=404)
    msgs = session.exec(
        select(Message).where(Message.thread_id == thread_id)
    ).all()
    return [{"id": m.id, "role": m.role, "content_markdown": m.content_markdown,
             "abstained": m.abstained, "cost_eur": m.cost_eur,
             "status": m.status, "current_stage": m.current_stage,
             "created_at": m.created_at, "cached": m.cached} for m in msgs]


@router.patch("/threads/{thread_id}")
def rename_thread(thread_id: int, body: ThreadRename, request: Request,
                  user: User = Depends(get_current_active_user),
                  session: Session = Depends(get_db_session)):
    thread = session.get(Thread, thread_id)
    if not thread or thread.user_id != user.id:
        raise HTTPException(status_code=404)
    thread.title = body.title.strip() or thread.title
    session.add(thread)
    session.commit()
    return {"id": thread.id, "title": thread.title}


@router.delete("/threads/{thread_id}", status_code=204)
def delete_thread(thread_id: int, request: Request,
                  user: User = Depends(get_current_active_user),
                  session: Session = Depends(get_db_session)):
    thread = session.get(Thread, thread_id)
    if not thread or thread.user_id != user.id:
        raise HTTPException(status_code=404)
    msg_ids = [m.id for m in session.exec(
        select(Message).where(Message.thread_id == thread_id)
    ).all()]
    if msg_ids:
        session.exec(sql_delete(Trace).where(Trace.message_id.in_(msg_ids)))
        session.exec(sql_delete(Message).where(Message.thread_id == thread_id))
    session.delete(thread)
    session.commit()


# ---------- ask ----------

@router.post("/threads/{thread_id}/ask")
def ask(thread_id: int, body: AskRequest, request: Request,
        background_tasks: BackgroundTasks,
        user: User = Depends(get_current_active_user),
        session: Session = Depends(get_db_session)):
    import logging
    _log = logging.getLogger(__name__)

    thread = session.get(Thread, thread_id)
    if not thread or thread.user_id != user.id:
        raise HTTPException(status_code=404)

    check_budget(session, user)

    version_tag: str = getattr(request.app.state, "version_tag", "dev")

    # Cache lookup (synchronous, before background task)
    from webapp.backend.cache import norm_query as _norm, make_query_hash
    _nq = _norm(body.query)
    _qhash = make_query_hash(_nq, version_tag)
    cache_entry = session.get(AnswerCache, _qhash)

    now = datetime.now(timezone.utc).isoformat()
    user_msg = Message(thread_id=thread_id, user_id=user.id, role="user",
                       content_markdown=body.query, status="done", created_at=now)
    session.add(user_msg)

    if cache_entry is not None:
        # HIT: build done response from cache, no RAG needed
        _log.info("CACHE HIT query_hash=%s", _qhash)
        asst_msg = Message(
            thread_id=thread_id, user_id=user.id, role="assistant",
            status="done", cached=True, current_stage="done",
            content_markdown=cache_entry.answer_markdown,
            abstained=bool(cache_entry.abstained),
            cost_eur=0.0, created_at=now,
        )
        session.add(asst_msg)
        session.commit()
        session.refresh(asst_msg)

        trace_data = json.loads(cache_entry.trace_json or "{}")
        trace = Trace(
            message_id=asst_msg.id,
            steps_json=json.dumps(trace_data.get("steps", [])),
            total_cost_eur=0.0,
            total_duration_ms=0,
            retrieved_doc_ids=json.dumps(trace_data.get("retrieved_doc_ids", [])),
            used_brute_force=bool(trace_data.get("used_brute_force", False)),
            abstained=bool(cache_entry.abstained),
            critic_detail_json=json.dumps(trace_data.get("critic_detail")),
            chunks_json=json.dumps(trace_data.get("chunks", [])),
            cited_sources_json=json.dumps(trace_data.get("cited_sources", [])),
            query_expansion_detail_json=json.dumps(trace_data.get("query_expansion_detail")),
            generator_detail_json=json.dumps(trace_data.get("generator_detail")),
        )
        session.add(trace)
        cache_entry.hit_count += 1
        session.add(cache_entry)
        session.commit()
        return {"message_id": asst_msg.id}

    # MISS: existing path
    _log.info("CACHE MISS query_hash=%s", _qhash)
    asst_msg = Message(thread_id=thread_id, user_id=user.id, role="assistant",
                       status="pending", current_stage="queued", created_at=now)
    session.add(asst_msg)
    session.commit()
    session.refresh(asst_msg)

    background_tasks.add_task(
        run_rag_in_background,
        engine=request.app.state.engine,
        rag=getattr(request.app.state, "rag", None),
        message_id=asst_msg.id,
        query=body.query,
        user_id=user.id,
        version_tag=version_tag,
    )
    return {"message_id": asst_msg.id}


def run_rag_in_background(
    engine, rag, message_id: int, query: str, user_id: int,
    version_tag: str = "dev",
) -> None:
    """Runs RAG synchronously in a thread, updates DB with result."""
    from webapp.backend.db import Message, Trace, User, Settings, AnswerCache
    from webapp.backend.trace_collector import TraceCollector, set_active_collector
    import logging
    _log = logging.getLogger(__name__)

    import os
    usd_eur_rate = float(os.getenv("USD_EUR_RATE", "0.92"))

    def _update_stage(stage: str) -> None:
        with Session(engine) as s:
            m = s.get(Message, message_id)
            if m:
                m.current_stage = stage
                s.add(m)
                s.commit()

    with Session(engine) as session:
        msg = session.get(Message, message_id)
        if not msg:
            return
        msg.status = "running"
        session.add(msg)
        session.commit()

        collector = TraceCollector()
        set_active_collector(collector)

        try:
            if rag is None:
                raise RuntimeError("RAG not initialized")
            import time as _time
            _t0 = _time.perf_counter()
            answer = rag.run(query, stage_callback=_update_stage)
            total_ms = int((_time.perf_counter() - _t0) * 1000)
            total_cost = collector.total_cost_eur

            msg.content_markdown = answer.answer_markdown
            msg.abstained = answer.abstained
            msg.cost_eur = total_cost
            msg.status = "done"
            msg.current_stage = "done"
            session.add(msg)

            def _step_dict(s):
                d = {k: v for k, v in s.__dict__.items() if not k.startswith("_")}
                # attach stage marker info
                stage = collector.stage_markers
                d["stage_markers"] = stage
                return d

            # kcsp FinalAnswer fields vs dkv audit_metadata
            _gen_step = answer.audit_metadata.get("generator_step") or {}
            generator_detail = {
                "generator_confidence": answer.audit_metadata.get("generator_confidence"),
                "generator_cot": answer.audit_metadata.get("generator_cot"),
                "model": _gen_step.get("model"),
                "tokens_prompt": _gen_step.get("tokens_prompt"),
                "tokens_completion": _gen_step.get("tokens_completion"),
                "cost_eur": _gen_step.get("cost_eur"),
                "duration_ms": _gen_step.get("duration_ms"),
            }
            _critic_step = answer.audit_metadata.get("critic_step") or {}
            critic_step_detail = {
                "model": _critic_step.get("model"),
                "tokens_prompt": _critic_step.get("tokens_prompt"),
                "tokens_completion": _critic_step.get("tokens_completion"),
                "cost_eur": _critic_step.get("cost_eur"),
                "duration_ms": _critic_step.get("duration_ms"),
            }
            critic_detail = {
                k: answer.audit_metadata.get(k)
                for k in ("critic_verdict", "critic_reasoning", "critic_cot",
                          "critic_confidence", "abstain_reason", "retried",
                          "used_ensemble", "early_abstain")
            }
            critic_detail.update(critic_step_detail)
            selector_detail = {
                k: answer.audit_metadata.get(k)
                for k in ("selector_confidence", "selector_cot")
            }
            pruning_detail = {
                **(answer.audit_metadata.get("pruning_detail") or {}),
                "detected_tarif": answer.audit_metadata.get("detected_tarif"),
            }
            steps_list = [_step_dict(s) for s in collector.steps]
            chunks_list = answer.audit_metadata.get("selected_chunks") or []
            cited_list = answer.cited_sources or []
            query_expansion_detail = collector.query_expansion_detail
            trace = Trace(
                message_id=message_id,
                steps_json=json.dumps(steps_list),
                total_cost_eur=total_cost,
                total_duration_ms=total_ms,
                retrieved_doc_ids=json.dumps(answer.retrieved_doc_ids),
                used_brute_force=answer.used_brute_force,
                abstained=answer.abstained,
                critic_detail_json=json.dumps(critic_detail),
                chunks_json=json.dumps(chunks_list),
                cited_sources_json=json.dumps(cited_list),
                query_expansion_detail_json=json.dumps(query_expansion_detail),
                selector_detail_json=json.dumps(selector_detail),
                pruning_detail_json=json.dumps(pruning_detail),
                generator_detail_json=json.dumps(generator_detail),
            )
            session.add(trace)

            # Save to answer_cache (INSERT OR IGNORE so parallel requests don't conflict)
            from webapp.backend.cache import norm_query as _norm, make_query_hash
            from datetime import datetime, timezone as _tz
            _nq = _norm(query)
            _qhash = make_query_hash(_nq, version_tag)
            if session.get(AnswerCache, _qhash) is None:
                _trace_payload = {
                    "steps": steps_list,
                    "total_cost_eur": total_cost,
                    "total_duration_ms": total_ms,
                    "retrieved_doc_ids": answer.retrieved_doc_ids,
                    "used_brute_force": answer.used_brute_force,
                    "abstained": answer.abstained,
                    "critic_detail": critic_detail,
                    "chunks": chunks_list,
                    "cited_sources": cited_list,
                    "query_expansion_detail": query_expansion_detail,
                    "selector_detail": selector_detail,
                    "pruning_detail": pruning_detail,
                    "generator_detail": generator_detail,
                }
                session.add(AnswerCache(
                    query_hash=_qhash,
                    normalized_query=_nq,
                    version_tag=version_tag,
                    answer_markdown=answer.answer_markdown,
                    abstained=bool(answer.abstained),
                    trace_json=json.dumps(_trace_payload),
                    created_at=datetime.now(_tz.utc).isoformat(),
                ))

            session.commit()  # commit msg + trace + cache atomically
            user = session.get(User, user_id)
            if user:
                deduct_cost(session, user, total_cost)

        except Exception as exc:
            msg.status = "error"
            msg.current_stage = "error"
            msg.content_markdown = f"Error: {exc}"
            session.add(msg)
            session.commit()
        finally:
            set_active_collector(None)


# ---------- status + SSE stream + trace ----------

@router.get("/chat/{message_id}/status")
def message_status(message_id: int, request: Request,
                   user: User = Depends(get_current_active_user),
                   session: Session = Depends(get_db_session)):
    msg = session.get(Message, message_id)
    if not msg or msg.user_id != user.id:
        raise HTTPException(status_code=404)
    return {"status": msg.status, "stage": msg.current_stage}


@router.get("/chat/{message_id}/stream")
async def stream_message(
    message_id: int,
    request: Request,
    user: User = Depends(get_current_active_user),
):
    engine = request.app.state.engine
    with Session(engine) as s:
        msg = s.get(Message, message_id)
        if not msg or msg.user_id != user.id:
            raise HTTPException(status_code=404)

    async def _generate():
        last_stage: Optional[str] = None
        while True:
            with Session(engine) as s:
                m = s.get(Message, message_id)
                if m is None:
                    break
                stage = m.current_stage
                if stage and stage not in ("done", "error", "queued") and stage != last_stage:
                    last_stage = stage
                    yield f"event: stage\ndata: {json.dumps({'stage': stage})}\n\n"
                if m.status in ("done", "error"):
                    payload = json.dumps({
                        "message_id": message_id,
                        "status": m.status,
                        "abstained": m.abstained,
                    })
                    yield f"event: done\ndata: {payload}\n\n"
                    return
            await asyncio.sleep(0.25)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_trace_payload(message_id: int, session: Session) -> dict:
    trace = session.exec(select(Trace).where(Trace.message_id == message_id)).first()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not available yet")

    raw_critic = json.loads(trace.critic_detail_json or "null") or {}
    pruning = json.loads(trace.pruning_detail_json or "{}") or {}
    raw_gen = json.loads(trace.generator_detail_json or "null") or {}
    qe = json.loads(trace.query_expansion_detail_json or "null")

    critic = {
        "verdict": raw_critic.get("critic_verdict"),
        "confidence": raw_critic.get("critic_confidence"),
        "reasoning": raw_critic.get("critic_reasoning") or [],
        "chain_of_thought": raw_critic.get("critic_cot") or [],
        "retried": raw_critic.get("retried"),
        "used_ensemble": raw_critic.get("used_ensemble"),
        "model": raw_critic.get("model"),
        "tokens_prompt": raw_critic.get("tokens_prompt"),
        "tokens_completion": raw_critic.get("tokens_completion"),
        "cost_eur": raw_critic.get("cost_eur"),
        "duration_ms": raw_critic.get("duration_ms"),
    } if raw_critic else None

    retrieval = {
        "detected_tarif": pruning.get("detected_tarif"),
        "chunks": json.loads(trace.chunks_json or "[]"),
    }

    generator = {
        "model": raw_gen.get("model"),
        "tokens_prompt": raw_gen.get("tokens_prompt"),
        "tokens_completion": raw_gen.get("tokens_completion"),
        "cost_eur": raw_gen.get("cost_eur"),
        "duration_ms": raw_gen.get("duration_ms"),
        "confidence": raw_gen.get("generator_confidence"),
        "chain_of_thought": raw_gen.get("generator_cot") or [],
    } if raw_gen else None

    # Flatten QueryExpansion step stats to top level (stored nested under "step")
    if qe and isinstance(qe, dict):
        step = qe.pop("step", None) or {}
        for key in ("model", "tokens_prompt", "tokens_completion", "cost_eur", "duration_ms"):
            if qe.get(key) is None:
                qe[key] = step.get(key)

    return {
        "total_cost_eur": trace.total_cost_eur,
        "total_duration_ms": trace.total_duration_ms,
        "abstained": trace.abstained,
        "cited_sources": json.loads(trace.cited_sources_json or "[]"),
        "query_expansion": qe,
        "retrieval": retrieval,
        "generator": generator,
        "critic": critic,
    }


@router.get("/messages/{message_id}/trace")
def get_trace(message_id: int, request: Request,
              user: User = Depends(get_current_active_user),
              session: Session = Depends(get_db_session)):
    msg = session.get(Message, message_id)
    if not msg or msg.user_id != user.id:
        raise HTTPException(status_code=404)
    return _build_trace_payload(message_id, session)
