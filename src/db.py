from __future__ import annotations
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def get_engine():
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pw = os.getenv("PG_PASSWORD", "")
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}", future=True)

def schema_name() -> str:
    return os.getenv("PG_SCHEMA", "pricing")