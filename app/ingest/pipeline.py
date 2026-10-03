"""Ingest saved warning letters: parse -> extract -> chunk -> embed -> Postgres + Neo4j.

    python -m app.ingest.pipeline              # all letters in data/raw (max 15)
    python -m app.ingest.pipeline --refresh    # ignore cached LLM extractions
    python -m app.ingest.pipeline --doc <id>   # one letter
"""
import argparse
import logging

from app.graph import load as graph
from app.ingest import store
from app.ingest.chunk import chunk
from app.ingest.extract import extract
from app.ingest.fda_scraper import MAX_DOCS, RAW_DIR
from app.ingest.parse import parse
from app.retrieval.embeddings import embed_passages


def run(doc_ids: list[str] | None = None, refresh: bool = False) -> None:
    paths = sorted(RAW_DIR.glob("*.html"))[:MAX_DOCS]
    if doc_ids:
        paths = [p for p in paths if p.stem in doc_ids]
    if not paths:
        raise SystemExit(f"no matching HTML in {RAW_DIR} — run app.ingest.fda_scraper first")

    with store.connect() as pg, graph.driver() as neo:
        graph.ensure_schema(neo)
        for path in paths:
            doc = parse(path)
            ex = extract(doc, refresh=refresh)
            obs = ex.extraction.observations
            obs_ids = [f"{doc.meta['id']}:obs{k:02d}" for k in range(len(obs))]
            chunks = chunk(doc, obs, obs_ids)
            vectors = embed_passages([c.embed_text for c in chunks])

            store.upsert_document(pg, doc, str(path.relative_to(RAW_DIR.parents[1])))
            store.replace_chunks(pg, doc.meta["id"], chunks, vectors)
            graph.load_document(neo, doc, ex, chunks, obs_ids)
            pg.commit()
            linked = sum(c.observation_id is not None for c in chunks)
            print(f"{doc.meta['id']}: {len(obs)} observations, {len(chunks)} chunks "
                  f"({linked} linked to observations)")
    print(f"done: {len(paths)} documents")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", help="document id (repeatable)")
    ap.add_argument("--refresh", action="store_true", help="re-run LLM extraction")
    args = ap.parse_args()
    run(args.doc, args.refresh)
