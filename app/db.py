"""Shared connection pools for the API process."""
from functools import lru_cache

from neo4j import Driver
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings
from app.graph.load import driver as neo4j_driver


@lru_cache
def pg_pool() -> ConnectionPool:
    return ConnectionPool(settings.postgres_dsn, min_size=1, max_size=5, configure=register_vector, open=True)


@lru_cache
def neo4j() -> Driver:
    return neo4j_driver()
