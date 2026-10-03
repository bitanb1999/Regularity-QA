"""Schema the LLM must fill for each warning letter (sent to Groq as a strict JSON schema)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Topic = Literal[
    "data_integrity",
    "laboratory_controls",
    "oos_investigation",
    "quality_unit",
    "component_testing",
    "process_validation",
    "cleaning_validation",
    "stability_testing",
    "storage_conditions",
    "contamination_control",
    "pathogen_contamination",
    "sanitation_hygiene",
    "equipment_facilities",
    "hazard_analysis",
    "supplier_verification",
    "labeling_misbranding",
    "unapproved_drug",
    "false_advertising",
    "registration_listing",
    "other",
]

InspectionType = Literal[
    "on_site_inspection", "records_review", "sample_analysis", "website_review", "other"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SiteX(_Strict):
    name: str = Field(description="Name of the facility FDA inspected or reviewed")
    address: str | None = Field(description="Facility street address as written, or null")
    fei: str | None = Field(description="FDA Establishment Identifier (FEI) digits, or null")


class InspectionX(_Strict):
    type: InspectionType = Field(
        description="How FDA found the violations. Any in-person inspection (CGMP, FSVP, HACCP) "
        "is on_site_inspection; review of submitted records is records_review"
    )
    start_date: str | None = Field(description="First inspection/review date, YYYY-MM-DD, or null")
    end_date: str | None = Field(description="Last inspection/review date, YYYY-MM-DD, or null")
    form_483_issued: bool = Field(description="True if the letter says a Form FDA 483/483a was issued")


class ObservationX(_Strict):
    number: int | None = Field(description="The number FDA printed for this violation, or null if unnumbered")
    section: str | None = Field(description="Heading the violation sits under, e.g. 'Misbranding Violations', or null")
    title: str = Field(description="Short label for the violation, at most 12 words")
    summary: str = Field(description="1-2 factual sentences on what FDA found; no speculation")
    first_block: int = Field(description="Index of the block where this violation starts")
    last_block: int = Field(description="Index of the last block belonging to this violation")
    regulations: list[str] = Field(
        description="Regulations FDA says were violated, formatted '21 CFR 211.192(b)', "
        "'21 CFR Part 117' or 'FD&C Act 501(a)(2)(B)'"
    )
    topics: list[Topic] = Field(description="1-3 topics from the allowed list")
    is_repeat: bool = Field(description="True only if the letter says it was also cited at a previous inspection")


class Extraction(_Strict):
    site: SiteX
    inspection: InspectionX
    observations: list[ObservationX]


class GapFill(_Strict):
    observations: list[ObservationX]
