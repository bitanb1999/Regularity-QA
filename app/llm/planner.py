"""Question router: classify a question and fill (at most) one whitelisted graph call.

The LLM's only interface to Neo4j is this schema: a template name from a fixed list plus typed
parameters. app.graph.queries validates it again before anything runs.
"""
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.graph.queries import Template
from app.llm.client import Usage, structured
from app.models.extraction import Topic

log = logging.getLogger(__name__)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphCallX(_Strict):
    template: Template
    topic: Topic | None = Field(description="Required for observations_by_topic, else null")
    regulation: str | None = Field(
        description="Required for observations_by_regulation: '21 CFR 211.192', '21 CFR Part 211' or "
        "'FD&C Act 501(a)'. Else null"
    )
    company: str | None = Field(description="Company name fragment to filter by, or null")
    start_date: str | None = Field(description="YYYY-MM-DD lower bound on letter issue date, or null")
    end_date: str | None = Field(description="YYYY-MM-DD upper bound on letter issue date, or null")


class Plan(_Strict):
    route: Literal["semantic", "structural", "both"]
    search_query: str = Field(description="The question rewritten as a standalone search query")
    company: str | None = Field(description="If the question is about one company, its name; else null")
    graph: GraphCallX | None = Field(description="Graph call for structural/both routes, else null")


SYSTEM = """You route questions for a Q&A system over FDA warning letters.

Two retrieval tools exist:
- semantic search over letter text: for "what / how / why / describe" questions about content.
- a knowledge graph with fixed query templates: for listing, filtering, counting and comparing
  companies, sites, observations, regulations and topics.

route: "semantic" (text search only), "structural" (graph only), or "both" (a graph filter plus
text detail, e.g. "what did FDA say about data integrity at companies cited under 21 CFR 211?").
When unsure, prefer "both".

Graph templates:
- observations_by_topic(topic, company?): observations in a topic
- observations_by_regulation(regulation, company?): observations citing a CFR section, CFR part,
  or FD&C Act section (parts include all their sections)
- observations_by_company(company): everything cited against one company
- repeat_observations(company?): violations FDA says were also cited at a previous inspection
- observations_in_period(start_date?, end_date?, company?): letters issued in a date range
- top_regulations(company?): regulations cited most often
- topic_counts(company?): how often each topic is cited

Topics: data_integrity, laboratory_controls, oos_investigation, quality_unit, component_testing,
process_validation, cleaning_validation, stability_testing, storage_conditions,
contamination_control, pathogen_contamination, sanitation_hygiene, equipment_facilities,
hazard_analysis, supplier_verification, labeling_misbranding, unapproved_drug, false_advertising,
registration_listing, other.

Never invent template names. If no template fits, set graph to null and use "semantic"."""


def plan(question: str, usage: Usage | None = None) -> tuple[Plan, str | None]:
    """Returns (plan, error). Falls back to plain semantic search if the router fails."""
    try:
        p = structured(SYSTEM, question, Plan, model=settings.router_model, max_tokens=800, usage=usage)
        return p, None
    except Exception as e:  # noqa: BLE001 - router failure must not take the whole request down
        log.warning("planner failed, falling back to semantic: %s", e)
        fallback = Plan(route="semantic", search_query=question, company=None, graph=None)
        return fallback, f"planner_failed: {type(e).__name__}"
