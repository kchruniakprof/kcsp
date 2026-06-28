"""Database models and initialization for KCSP webapp."""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str = Field(default="")
    role: str = Field(default="user")       # 'user' | 'admin'
    status: str = Field(default="pending")  # 'pending' | 'active' | 'blocked'
    budget_eur: float = Field(default=0.50)
    spent_eur: float = Field(default=0.0)
    otp_verified: bool = Field(default=False)
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None


class EmailOtp(SQLModel, table=True):
    __tablename__ = "email_otps"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    code_hash: str
    created_at: str
    attempts: int = Field(default=0)
    verified: bool = Field(default=False)


class Thread(SQLModel, table=True):
    __tablename__ = "threads"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    thread_id: int = Field(foreign_key="threads.id")
    user_id: int = Field(foreign_key="users.id")
    role: str                               # 'user' | 'assistant'
    content_markdown: Optional[str] = None
    abstained: bool = Field(default=False)
    cost_eur: Optional[float] = None
    status: str = Field(default="pending")  # 'pending'|'running'|'done'|'error'|'timeout'
    current_stage: Optional[str] = None
    created_at: Optional[str] = None
    cached: bool = Field(default=False)     # True when answer served from answer_cache


class Trace(SQLModel, table=True):
    __tablename__ = "traces"

    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="messages.id", unique=True)
    steps_json: Optional[str] = None        # JSON array of StepTrace
    total_cost_eur: Optional[float] = None
    total_duration_ms: Optional[int] = None
    retrieved_doc_ids: Optional[str] = None # JSON array
    used_brute_force: bool = Field(default=False)
    abstained: bool = Field(default=False)
    critic_detail_json: Optional[str] = None  # JSON: {critic_verdict, reasoning, cot, confidence, abstain_reason}
    chunks_json: Optional[str] = None  # JSON array of selected source chunks (text + curated metadata)
    cited_sources_json: Optional[str] = None  # JSON array of chunk_ids the generator actually cited
    query_expansion_detail_json: Optional[str] = None  # JSON: {intent, paraphrases, domain_terms, section_types, chain_of_thought, confidence_score}
    selector_detail_json: Optional[str] = None  # JSON: {confidence, chain_of_thought}
    pruning_detail_json: Optional[str] = None   # JSON: {strategy, chars_before, chars_after, sentences_dropped}
    generator_detail_json: Optional[str] = None  # JSON: {confidence, cot}


class AnswerCache(SQLModel, table=True):
    __tablename__ = "answer_cache"

    query_hash: str = Field(primary_key=True)       # sha256(norm_query + NUL + version_tag)
    normalized_query: str = Field(default="")       # for diagnostics
    version_tag: str = Field(default="")
    answer_markdown: str = Field(default="")
    abstained: bool = Field(default=False)
    trace_json: Optional[str] = None                # JSON payload matching GET /messages/{id}/trace
    created_at: Optional[str] = None
    hit_count: int = Field(default=0)


class OAuthState(SQLModel, table=True):
    __tablename__ = "oauth_states"

    state: str = Field(primary_key=True)
    nonce: Optional[str] = None
    created_at: str = Field(default="")


class Settings(SQLModel, table=True):
    __tablename__ = "settings"

    id: int = Field(default=1, primary_key=True)
    default_budget_eur: float = Field(default=0.50)
    usd_eur_rate: float = Field(default=0.92)
    global_daily_limit_eur: float = Field(default=10.0)
    spent_today_eur: float = Field(default=0.0)
    today_date: Optional[str] = None
    kill_switch: bool = Field(default=False)


def create_db_and_tables(engine) -> None:
    SQLModel.metadata.create_all(engine)


def run_migrations(engine) -> None:
    """Additive schema migrations for existing databases."""
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE traces ADD COLUMN critic_detail_json TEXT",
        "ALTER TABLE traces ADD COLUMN chunks_json TEXT",
        "ALTER TABLE traces ADD COLUMN cited_sources_json TEXT",
        "ALTER TABLE traces ADD COLUMN query_expansion_detail_json TEXT",
        "ALTER TABLE traces ADD COLUMN selector_detail_json TEXT",
        "ALTER TABLE traces ADD COLUMN pruning_detail_json TEXT",
        "ALTER TABLE traces ADD COLUMN generator_detail_json TEXT",
        "ALTER TABLE messages ADD COLUMN cached INTEGER DEFAULT 0",
        (
            "CREATE TABLE IF NOT EXISTS oauth_states ("
            "state TEXT PRIMARY KEY, nonce TEXT, created_at TEXT)"
        ),
        (
            "CREATE TABLE IF NOT EXISTS answer_cache ("
            "query_hash TEXT PRIMARY KEY, normalized_query TEXT, version_tag TEXT, "
            "answer_markdown TEXT, abstained INTEGER DEFAULT 0, trace_json TEXT, "
            "created_at TEXT, hit_count INTEGER DEFAULT 0)"
        ),
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                pass  # column already exists


def seed_settings(session: Session) -> None:
    existing = session.exec(select(Settings).where(Settings.id == 1)).first()
    if existing is None:
        session.add(Settings(id=1, today_date=str(date.today())))
        session.commit()


def get_engine(db_path: str = ":memory:"):
    from sqlalchemy.pool import StaticPool
    if db_path == ":memory:":
        return create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
