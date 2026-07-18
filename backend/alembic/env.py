from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Project root on sys.path so app.db.models is importable ───────────────────
_backend_dir = Path(__file__).parent.parent          # .../backend
_project_root = _backend_dir.parent                  # .../GurriculumGPT
sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_project_root))

# ── Load DATABASE_URL from .env without going through pydantic-settings ───────
_env_file = _project_root / ".env"
if _env_file.exists():
    for _raw in _env_file.read_text(encoding="utf-8").splitlines():
        _raw = _raw.strip()
        if _raw and not _raw.startswith("#") and "=" in _raw:
            _k, _, _v = _raw.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Import ORM metadata (Base must be populated before autogenerate works) ────
from app.database import Base          # noqa: E402
import app.db.models                   # noqa: F401, E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Use DATABASE_URL env var (set above from .env, or injected by shell)
_db_url = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://gurriculum:password@127.0.0.1:5432/gurriculum_db",
)
config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
