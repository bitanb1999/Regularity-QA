"""Postgres storage: `documents` (one row per letter) and `chunks` (pgvector + full-text)."""
import psycopg
from pgvector.psycopg import register_vector

from app.config import settings
from app.ingest.chunk import Chunk
from app.ingest.parse import ParsedDoc

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    doc_type       TEXT NOT NULL,
    url            TEXT UNIQUE NOT NULL,
    title          TEXT NOT NULL,
    company        TEXT,
    cms_number     TEXT,
    issue_date     DATE,
    subject        TEXT,
    product        TEXT,
    issuing_office TEXT,
    city           TEXT,
    region         TEXT,
    country        TEXT,
    body           TEXT NOT NULL,
    raw_path       TEXT NOT NULL,
    stored_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS street TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS postal_code TEXT;

CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal        INT NOT NULL,
    section        TEXT NOT NULL,
    observation_id TEXT,
    block_start    INT NOT NULL,
    block_end      INT NOT NULL,
    text           TEXT NOT NULL,
    embedding      vector({settings.embedding_dim}) NOT NULL,
    tsv            tsvector GENERATED ALWAYS AS (
                       setweight(to_tsvector('english', section), 'A') ||
                       setweight(to_tsvector('english', text), 'B')) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""

UPSERT_DOC = """
INSERT INTO documents (id, doc_type, url, title, company, cms_number, issue_date, subject, product,
                       issuing_office, street, city, region, postal_code, country, body, raw_path)
VALUES (%(id)s, %(doc_type)s, %(url)s, %(title)s, %(company)s, %(cms_number)s, %(issue_date)s,
        %(subject)s, %(product)s, %(issuing_office)s, %(street)s, %(city)s, %(region)s,
        %(postal_code)s, %(country)s, %(body)s, %(raw_path)s)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title, company = EXCLUDED.company, cms_number = EXCLUDED.cms_number,
    issue_date = EXCLUDED.issue_date, subject = EXCLUDED.subject, product = EXCLUDED.product,
    issuing_office = EXCLUDED.issuing_office, street = EXCLUDED.street, city = EXCLUDED.city,
    region = EXCLUDED.region, postal_code = EXCLUDED.postal_code, country = EXCLUDED.country,
    body = EXCLUDED.body, raw_path = EXCLUDED.raw_path, stored_at = now();
"""

INSERT_CHUNK = """
INSERT INTO chunks (id, doc_id, ordinal, section, observation_id, block_start, block_end, text, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.postgres_dsn)
    conn.execute(SCHEMA)
    register_vector(conn)
    conn.commit()
    return conn


def upsert_document(conn: psycopg.Connection, doc: ParsedDoc, raw_path: str) -> None:
    conn.execute(UPSERT_DOC, {**doc.meta, "body": doc.body, "raw_path": raw_path})


def replace_chunks(
    conn: psycopg.Connection, doc_id: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    import numpy as np

    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    with conn.cursor() as cur:
        cur.executemany(INSERT_CHUNK, [
            (c.id, c.doc_id, c.ordinal, c.section, c.observation_id, c.block_start, c.block_end,
             c.text, np.asarray(v, dtype=np.float32))
            for c, v in zip(chunks, vectors)
        ])
