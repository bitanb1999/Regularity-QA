"""Whitelisted, parameterized Cypher templates — the only way the LLM can query the graph.

The planner picks a template name and fills typed parameters; it never writes Cypher. Queries
are assembled solely from the constant fragments below, values are passed as bound parameters,
and everything runs in a read-only transaction with a timeout and a row limit.
"""
from dataclasses import dataclass
from typing import Literal, get_args

from neo4j import Driver, ManagedTransaction, unit_of_work

from app.ingest.citations import canonical
from app.models.extraction import Topic

Template = Literal[
    "observations_by_topic",
    "observations_by_regulation",
    "observations_by_company",
    "repeat_observations",
    "observations_in_period",
    "top_regulations",
    "topic_counts",
]
TEMPLATES: tuple[str, ...] = get_args(Template)
TOPICS: frozenset[str] = frozenset(get_args(Topic))

TIMEOUT_S = 5.0
MAX_ROWS = 50

_OBS_MATCH = """
MATCH (c:Company)-[:OWNS]->(s:Site)-[:INSPECTED_IN]->(i:Inspection)-[:HAS_OBSERVATION]->(o:Observation),
      (d:WarningLetter)-[:BASED_ON]->(i)
WHERE ($company IS NULL OR toLower(c.name) CONTAINS toLower($company))
"""

_OBS_RETURN = """
OPTIONAL MATCH (o)-[:CITES]->(r:Regulation)
OPTIONAL MATCH (o)-[:CATEGORY]->(t:Topic)
OPTIONAL MATCH (o)-[:EVIDENCED_BY]->(ch:Chunk)
WITH c, s, i, o, d, collect(DISTINCT r.id) AS regulations, collect(DISTINCT t.name) AS topics,
     collect(DISTINCT ch.id) AS chunk_ids
RETURN c.name AS company, s.name AS site, s.fei AS fei, d.id AS doc_id,
       toString(d.issue_date) AS issue_date, toString(i.start_date) AS inspection_start,
       o.id AS observation_id, o.number AS number, o.section AS section, o.title AS title,
       o.summary AS summary, o.is_repeat AS is_repeat, regulations, topics, chunk_ids
ORDER BY d.issue_date DESC, o.id
LIMIT $limit
"""

_WHERE = {
    "observations_by_topic": "AND EXISTS { MATCH (o)-[:CATEGORY]->(:Topic {name: $topic}) }",
    "observations_by_regulation": (
        "AND EXISTS { MATCH (o)-[:CITES]->(r:Regulation) "
        "WHERE r.id = $regulation OR EXISTS { MATCH (r)-[:PART_OF]->(:Regulation {id: $regulation}) } }"
    ),
    "observations_by_company": "",
    "repeat_observations": "AND o.is_repeat = true",
    "observations_in_period": (
        "AND ($start_date IS NULL OR d.issue_date >= date($start_date)) "
        "AND ($end_date IS NULL OR d.issue_date <= date($end_date))"
    ),
}

_AGGREGATES = {
    # Regulations cited by the most observations, rolled up to CFR part / FD&C section if asked.
    "top_regulations": """
MATCH (c:Company)-[:OWNS]->(:Site)-[:INSPECTED_IN]->(:Inspection)-[:HAS_OBSERVATION]->(o:Observation)
      -[:CITES]->(r:Regulation)
WHERE $company IS NULL OR toLower(c.name) CONTAINS toLower($company)
OPTIONAL MATCH (r)-[:PART_OF]->(p:Regulation)
WITH CASE WHEN $rollup THEN coalesce(p.id, r.id) ELSE r.id END AS regulation, o, c
OPTIONAL MATCH (o)-[:EVIDENCED_BY]->(ch:Chunk)
WITH regulation, count(DISTINCT o) AS observations, collect(DISTINCT c.name) AS companies,
     collect(DISTINCT ch.id)[0..3] AS chunk_ids
RETURN regulation, observations, companies, chunk_ids
ORDER BY observations DESC, regulation
LIMIT $limit
""",
    "topic_counts": """
MATCH (c:Company)-[:OWNS]->(:Site)-[:INSPECTED_IN]->(:Inspection)-[:HAS_OBSERVATION]->(o:Observation)
      -[:CATEGORY]->(t:Topic)
WHERE $company IS NULL OR toLower(c.name) CONTAINS toLower($company)
OPTIONAL MATCH (o)-[:EVIDENCED_BY]->(ch:Chunk)
WITH t.name AS topic, count(DISTINCT o) AS observations, collect(DISTINCT c.name) AS companies,
     collect(DISTINCT ch.id)[0..3] AS chunk_ids
RETURN topic, observations, companies, chunk_ids
ORDER BY observations DESC, topic
LIMIT $limit
""",
}

AGGREGATE_TEMPLATES = frozenset(_AGGREGATES)

_REQUIRED = {
    "observations_by_topic": ("topic",),
    "observations_by_regulation": ("regulation",),
    "observations_by_company": ("company",),
}


_DESCRIBE = {
    "observations_by_topic": "observations about {topic}",
    "observations_by_regulation": "observations citing {regulation}",
    "observations_by_company": "observations for {company}",
    "repeat_observations": "observations FDA marks as repeats of a previous inspection",
    "observations_in_period": "letters issued between {start_date} and {end_date}",
    "top_regulations": "regulation citations",
    "topic_counts": "topic classifications",
}


class InvalidGraphCall(ValueError):
    pass


@dataclass
class GraphCall:
    template: str
    topic: str | None = None
    regulation: str | None = None
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    rollup: bool = True
    limit: int = 25

    def validated(self) -> "GraphCall":
        if self.template not in TEMPLATES:
            raise InvalidGraphCall(f"unknown template {self.template!r}")
        company = (self.company or "").strip() or None
        # Check required params after normalizing: a blank company must not become "all companies".
        present = {"topic": self.topic, "regulation": self.regulation, "company": company}
        for p in _REQUIRED.get(self.template, ()):
            if not present[p]:
                raise InvalidGraphCall(f"{self.template} requires {p}")
        if self.topic is not None and self.topic not in TOPICS:
            raise InvalidGraphCall(f"unknown topic {self.topic!r}")
        regulation = self.regulation
        if regulation is not None:
            regulation = canonical(regulation)
            if regulation is None:
                raise InvalidGraphCall(f"unrecognized regulation {self.regulation!r}")
        return GraphCall(
            self.template, self.topic, regulation, company, self.start_date, self.end_date,
            self.rollup, max(1, min(self.limit, MAX_ROWS)),
        )

    def describe(self) -> str:
        text = _DESCRIBE[self.template].format(**{k: v or "any" for k, v in self.params().items()})
        if self.company and self.template != "observations_by_company":
            text += f" at {self.company}"
        return text.replace("_", " ")

    def cypher(self) -> str:
        if self.template in _AGGREGATES:
            return _AGGREGATES[self.template]
        return _OBS_MATCH + _WHERE[self.template] + _OBS_RETURN

    def params(self) -> dict:
        return {
            "topic": self.topic, "regulation": self.regulation, "company": self.company,
            "start_date": self.start_date, "end_date": self.end_date,
            "rollup": self.rollup, "limit": self.limit,
        }


@unit_of_work(timeout=TIMEOUT_S)
def _read(tx: ManagedTransaction, cypher: str, params: dict) -> list[dict]:
    return [r.data() for r in tx.run(cypher, params)]


def run(drv: Driver, call: GraphCall) -> list[dict]:
    call = call.validated()
    with drv.session(default_access_mode="READ") as session:
        rows = session.execute_read(_read, call.cypher(), call.params())
    for row in rows:
        row["chunk_ids"] = sorted(row.get("chunk_ids") or [])
    return rows


DOCUMENT_OBSERVATIONS = """
MATCH (d:WarningLetter {id: $doc_id})-[:BASED_ON]->(:Inspection)-[:HAS_OBSERVATION]->(o:Observation)
OPTIONAL MATCH (o)-[:CITES]->(r:Regulation)
OPTIONAL MATCH (o)-[:CATEGORY]->(t:Topic)
OPTIONAL MATCH (o)-[:EVIDENCED_BY]->(ch:Chunk)
RETURN o.id AS observation_id, o.number AS number, o.section AS section, o.title AS title,
       o.summary AS summary, o.is_repeat AS is_repeat, collect(DISTINCT r.id) AS regulations,
       collect(DISTINCT t.name) AS topics, collect(DISTINCT ch.id) AS chunk_ids
ORDER BY o.id
"""


def document_observations(drv: Driver, doc_id: str) -> list[dict]:
    """Internal (not LLM-facing) lookup used by GET /documents/{id}."""
    with drv.session(default_access_mode="READ") as session:
        rows = session.execute_read(_read, DOCUMENT_OBSERVATIONS, {"doc_id": doc_id})
    for row in rows:
        row["chunk_ids"] = sorted(row["chunk_ids"])
        row["regulations"] = sorted(row["regulations"])
    return rows
