from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unibot.settings import (
    validate_postgres_direct_dsn,
    validate_postgres_pooled_dsn,
)

SessionFactory = Callable[[], Session]

DatabaseRole = Literal["direct", "pooled"]


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UNIBOT_",
        case_sensitive=False,
        extra="ignore",
    )

    postgres_direct_dsn: str = Field(..., min_length=1)
    postgres_pooled_dsn: str = Field(..., min_length=1)

    @field_validator("postgres_direct_dsn")
    @classmethod
    def validate_direct_dsn(cls, value: str) -> str:
        return validate_postgres_direct_dsn(value)

    @field_validator("postgres_pooled_dsn")
    @classmethod
    def validate_pooled_dsn(cls, value: str) -> str:
        return validate_postgres_pooled_dsn(value)


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


def _resolve_dsn(role: DatabaseRole) -> str:
    settings = get_database_settings()
    if role == "direct":
        return settings.postgres_direct_dsn
    return settings.postgres_pooled_dsn


def create_database_engine(role: DatabaseRole, dsn: str | None = None) -> Engine:
    return create_engine(
        dsn or _resolve_dsn(role),
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_direct_engine() -> Engine:
    return create_database_engine("direct")


@lru_cache(maxsize=1)
def get_runtime_engine() -> Engine:
    return create_database_engine("pooled")


def create_session_factory(bind: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=bind, autoflush=False, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_direct_session_factory() -> sessionmaker[Session]:
    return create_session_factory(get_direct_engine())


@lru_cache(maxsize=1)
def get_runtime_session_factory() -> sessionmaker[Session]:
    return create_session_factory(get_runtime_engine())


@contextmanager
def direct_session_scope() -> Iterator[Session]:
    session = get_direct_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def runtime_session_scope() -> Iterator[Session]:
    session = get_runtime_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
