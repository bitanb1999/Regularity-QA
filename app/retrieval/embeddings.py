"""Local sentence-transformers embeddings (bge-small, 384-d, cosine)."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

# bge models expect this instruction on queries only, not on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(text: str) -> list[float]:
    return model().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
