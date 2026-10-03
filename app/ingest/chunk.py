"""Section-aware chunking over parsed blocks.

Boundaries are forced at headings and at observation starts/ends, so every chunk belongs to at
most one observation and can serve as its citation. Blocks are never split.
"""
from dataclasses import dataclass

from app.ingest.parse import ParsedDoc
from app.models.extraction import ObservationX

TARGET_CHARS = 1200
MIN_UNOWNED_CHARS = 40  # drop stray date/salutation fragments that belong to no observation


@dataclass
class Chunk:
    id: str
    doc_id: str
    ordinal: int
    section: str
    observation_id: str | None
    block_start: int
    block_end: int
    text: str
    embed_text: str  # text plus a short context header, for the embedding model only


def chunk(doc: ParsedDoc, observations: list[ObservationX], obs_ids: list[str]) -> list[Chunk]:
    owner: dict[int, str] = {}
    for o, oid in zip(observations, obs_ids):
        for i in range(o.first_block, o.last_block + 1):
            owner[i] = oid

    doc_id, company = doc.meta["id"], doc.meta["company"] or doc.meta["title"]
    chunks: list[Chunk] = []
    buf: list = []
    section = "Introduction"

    def flush() -> None:
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        if owner.get(buf[0].idx) is None and len(text) < MIN_UNOWNED_CHARS:
            buf.clear()
            return
        n = len(chunks)
        chunks.append(Chunk(
            id=f"{doc_id}:{n:03d}",
            doc_id=doc_id,
            ordinal=n,
            section=section,
            observation_id=owner.get(buf[0].idx),
            block_start=buf[0].idx,
            block_end=buf[-1].idx,
            text=text,
            embed_text=f"{company} warning letter ({doc.meta['issue_date']}) — {section}\n\n{text}",
        ))
        buf.clear()

    for b in doc.blocks:
        if b.kind == "heading":
            flush()
            if not b.text.upper().startswith("WARNING LETTER"):  # letterhead, not a section
                section = b.text
            continue  # heading text lives in `section` / embed_text
        if b.kind == "footnote" and section != "Footnotes":
            flush()
            section = "Footnotes"
        if buf and (
            owner.get(b.idx) != owner.get(buf[-1].idx)
            or sum(len(x.text) for x in buf) + len(b.text) > TARGET_CHARS
        ):
            flush()
        buf.append(b)
    flush()
    return chunks
