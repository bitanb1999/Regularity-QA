"""Load one extracted warning letter into Neo4j.

(Company)-[:OWNS]->(Site)-[:INSPECTED_IN]->(Inspection)-[:HAS_OBSERVATION]->(Observation)
(Observation)-[:CITES]->(Regulation)-[:PART_OF]->(Regulation)
(Observation)-[:CATEGORY]->(Topic)
(Observation)-[:EVIDENCED_BY]->(Chunk)-[:FROM]->(Document:WarningLetter)
(WarningLetter)-[:ISSUED_TO]->(Company)
(WarningLetter)-[:BASED_ON]->(Inspection)
(WarningLetter)-[:CITES]->(Regulation)      // every citation anywhere in the letter

Chunk text lives in Postgres; Chunk nodes carry ids so graph answers can be cited.
Reloading a letter replaces its inspection/observation/chunk subgraph; shared nodes
(Company, Site, Regulation, Topic) are merged.
"""
import re

from neo4j import Driver, GraphDatabase, ManagedTransaction

from app.config import settings
from app.ingest.chunk import Chunk
from app.ingest.citations import find_citations, parent
from app.ingest.extract import ExtractedDoc
from app.ingest.parse import ParsedDoc

CONSTRAINTS = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (n:Company) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT site_id IF NOT EXISTS FOR (n:Site) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT inspection_id IF NOT EXISTS FOR (n:Inspection) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (n:Observation) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT regulation_id IF NOT EXISTS FOR (n:Regulation) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (n:Topic) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
]

CLEAR_DOC = """
MATCH (d:Document {id: $doc_id})
OPTIONAL MATCH (d)<-[:FROM]-(c:Chunk)
OPTIONAL MATCH (d)-[:BASED_ON]->(i:Inspection)
OPTIONAL MATCH (i)-[:HAS_OBSERVATION]->(o:Observation)
DETACH DELETE c, o, i
WITH d
OPTIONAL MATCH (d)-[r:CITES|ISSUED_TO]->()
DELETE r
"""

LOAD_DOC = """
MERGE (d:Document:WarningLetter {id: $doc.id})
SET d += $doc
MERGE (co:Company {name: $company})
MERGE (d)-[:ISSUED_TO]->(co)
MERGE (s:Site {id: $site.id})
SET s += $site
MERGE (co)-[:OWNS]->(s)
CREATE (i:Inspection {id: $insp.id, type: $insp.type, form_483_issued: $insp.form_483_issued,
                      start_date: CASE WHEN $insp.start_date IS NULL THEN null ELSE date($insp.start_date) END,
                      end_date: CASE WHEN $insp.end_date IS NULL THEN null ELSE date($insp.end_date) END})
CREATE (s)-[:INSPECTED_IN]->(i)
CREATE (d)-[:BASED_ON]->(i)
WITH d
UNWIND $doc_regs AS reg
MERGE (r:Regulation {id: reg.id}) SET r.framework = reg.framework
MERGE (d)-[:CITES]->(r)
"""

LOAD_REG_PARENTS = """
UNWIND $regs AS reg
MERGE (r:Regulation {id: reg.id}) SET r.framework = reg.framework
WITH r, reg WHERE reg.parent IS NOT NULL
MERGE (p:Regulation {id: reg.parent}) SET p.framework = reg.framework
MERGE (r)-[:PART_OF]->(p)
"""

LOAD_CHUNKS = """
MATCH (d:Document {id: $doc_id})
UNWIND $chunks AS ch
CREATE (c:Chunk {id: ch.id, ordinal: ch.ordinal, section: ch.section})
CREATE (c)-[:FROM]->(d)
"""

LOAD_OBSERVATIONS = """
MATCH (i:Inspection {id: $insp_id})
UNWIND $obs AS ob
CREATE (o:Observation {id: ob.id, number: ob.number, section: ob.section, title: ob.title,
                       summary: ob.summary, is_repeat: ob.is_repeat})
CREATE (i)-[:HAS_OBSERVATION]->(o)
FOREACH (reg IN ob.regulations | MERGE (r:Regulation {id: reg}) MERGE (o)-[:CITES]->(r))
FOREACH (t IN ob.topics | MERGE (tp:Topic {name: t}) MERGE (o)-[:CATEGORY]->(tp))
WITH o, ob
UNWIND ob.chunk_ids AS cid
MATCH (c:Chunk {id: cid})
CREATE (o)-[:EVIDENCED_BY]->(c)
"""


def driver() -> Driver:
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


def ensure_schema(drv: Driver) -> None:
    for stmt in CONSTRAINTS:
        drv.execute_query(stmt)


def _reg(reg_id: str) -> dict:
    return {"id": reg_id, "framework": "FD&C Act" if reg_id.startswith("FD&C") else "21 CFR",
            "parent": parent(reg_id)}


def _site_id(company: str, fei: str | None, address: str | None) -> str:
    if fei and (digits := re.sub(r"\D", "", fei)):
        return f"FEI:{digits}"
    return f"{company}|{(address or '').lower()}"


def _load(tx: ManagedTransaction, doc: ParsedDoc, ex: ExtractedDoc,
          chunks: list[Chunk], obs_ids: list[str]) -> None:
    m, x = doc.meta, ex.extraction
    doc_id = m["id"]
    company = m["company"] or x.site.name
    meta_addr = ", ".join(p for p in (m["street"], m["city"], m["region"], m["postal_code"]) if p)
    site = {
        "id": _site_id(company, x.site.fei, x.site.address or meta_addr),
        "name": x.site.name,
        "address": x.site.address or meta_addr,
        "fei": x.site.fei,
        "city": m["city"], "region": m["region"], "country": m["country"],
    }
    insp = {"id": f"{doc_id}:insp", **x.inspection.model_dump()}
    doc_props = {k: (str(v) if k == "issue_date" and v else v) for k, v in m.items()
                 if k in ("id", "title", "url", "cms_number", "issue_date", "subject", "product",
                          "issuing_office")}

    all_regs = sorted(find_citations(doc.body) | {r for o in x.observations for r in o.regulations})

    tx.run(CLEAR_DOC, doc_id=doc_id)
    tx.run(LOAD_REG_PARENTS, regs=[_reg(r) for r in all_regs])
    tx.run(LOAD_DOC, doc=doc_props, company=company, site=site, insp=insp,
           doc_regs=[_reg(r) for r in sorted(find_citations(doc.body))])
    tx.run("MATCH (d:Document {id: $id}) SET d.issue_date = date(d.issue_date)", id=doc_id)
    tx.run(LOAD_CHUNKS, doc_id=doc_id,
           chunks=[{"id": c.id, "ordinal": c.ordinal, "section": c.section} for c in chunks])
    tx.run(LOAD_OBSERVATIONS, insp_id=insp["id"], obs=[
        {**o.model_dump(include={"number", "section", "title", "summary", "is_repeat",
                                 "regulations", "topics"}),
         "id": oid,
         "chunk_ids": [c.id for c in chunks if c.observation_id == oid]}
        for o, oid in zip(x.observations, obs_ids)
    ])


def load_document(drv: Driver, doc: ParsedDoc, ex: ExtractedDoc,
                  chunks: list[Chunk], obs_ids: list[str]) -> None:
    with drv.session() as session:
        session.execute_write(_load, doc, ex, chunks, obs_ids)
