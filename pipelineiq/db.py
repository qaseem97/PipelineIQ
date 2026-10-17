"""Database tables shared by the state store and the process log.

Both tables are append-only: rows are inserted, never updated or deleted.
"""
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Integer, MetaData, String, Table, Text,
                        create_engine)

from . import config

metadata = MetaData()

# Every action ever dispatched. The current state is rebuilt by replaying these (event sourcing).
actions_table = Table(
    "actions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("project_id", String(64), index=True, nullable=False),
    Column("seq", Integer, nullable=False),          # per-project sequence = state version after this action
    Column("type", String(64), nullable=False),
    Column("agent", String(64), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# One row per Manager review: pass/fail, checks, decision and feedback.
process_log_table = Table(
    "process_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("project_id", String(64), index=True, nullable=False),
    Column("task_id", String(64), index=True, nullable=False),
    Column("phase", String(32), nullable=False),
    Column("agent", String(64), nullable=False),
    Column("attempt", Integer, nullable=False),
    Column("status", String(8), nullable=False),      # PASS | FAIL
    Column("score", Integer, nullable=True),
    Column("checks", JSON, nullable=False),
    Column("decision", String(16), nullable=False),   # APPROVE | REWORK | ESCALATE
    Column("feedback", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

_engine = None


def _create(url: str):
    args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=args, pool_pre_ping=True)


def get_engine(url: str | None = None):
    """Return a shared engine (or a fresh one when a URL is given, e.g. in tests)."""
    global _engine
    if url is not None:
        eng = _create(url)
        metadata.create_all(eng)
        return eng
    if _engine is None:
        _engine = _create(config.DATABASE_URL)
        metadata.create_all(_engine)
    return _engine


def now() -> datetime:
    return datetime.now(timezone.utc)
