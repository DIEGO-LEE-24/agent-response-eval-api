"""Public request/response contracts.

Every model forbids unknown fields so that a typo in a client payload fails
loudly (422) instead of being silently ignored.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Grounding check
# --------------------------------------------------------------------------- #


class EvidenceFact(StrictModel):
    """One structured fact the agent was allowed to rely on."""

    source_id: str = Field(min_length=1, max_length=120)
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=500)


class AnswerClaim(StrictModel):
    """One factual claim extracted from the answer by the caller.

    Claim extraction is *not* performed by this service. The rules evaluator
    only verifies the claims it is given.
    """

    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=500)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=20)


class HallucinationRequest(StrictModel):
    case_id: str = Field(min_length=1, max_length=80)
    answer: str = Field(max_length=8000)
    evidence: list[EvidenceFact] = Field(default_factory=list, max_length=100)
    claims: list[AnswerClaim] = Field(default_factory=list, max_length=100)
    mode: Literal["rules", "llm"] = "rules"


class Verdict(StrEnum):
    GROUNDED = "grounded"
    UNSUPPORTED = "unsupported"
    NEEDS_REVIEW = "needs_review"


class Violation(StrictModel):
    code: str
    claim_key: str | None = None
    detail: str


class HallucinationResponse(StrictModel):
    case_id: str
    verdict: Verdict
    violations: list[Violation]
    evaluator: str
    prompt_version: str | None = None
    human_review_required: bool
    checked_claims: int = Field(description="Number of structured claims that were verified")
    answer_text_assessed: bool = Field(
        description=(
            "False for the rules evaluator: it never reads the free-text answer beyond an "
            "emptiness check, so statements that were not extracted as claims are not judged."
        )
    )


# --------------------------------------------------------------------------- #
# Resolution check
# --------------------------------------------------------------------------- #


ToolActionStatus = Literal["not_required", "succeeded", "failed", "pending", "not_attempted"]


class ResolutionRequest(StrictModel):
    case_id: str = Field(min_length=1, max_length=80)
    customer_confirmed: bool | None = Field(
        default=None, description="Explicit customer confirmation; None means no signal"
    )
    requires_action: bool
    tool_action_status: ToolActionStatus = "not_required"
    handed_off: bool = False
    customer_goal_present: bool = True


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    HANDOFF = "handoff"
    UNKNOWN = "unknown"


class ResolutionResponse(StrictModel):
    case_id: str
    status: ResolutionStatus
    reason_code: str
    evaluator: str


# --------------------------------------------------------------------------- #
# Dataset evaluation
# --------------------------------------------------------------------------- #


class LabeledHallucinationCase(StrictModel):
    input: HallucinationRequest
    expected_verdict: Verdict
    category: str | None = Field(default=None, max_length=60)
    note: str | None = Field(default=None, max_length=300)


class LabeledResolutionCase(StrictModel):
    input: ResolutionRequest
    expected_status: ResolutionStatus
    category: str | None = Field(default=None, max_length=60)
    note: str | None = Field(default=None, max_length=300)


class EvaluationRequest(StrictModel):
    dataset_name: str = Field(min_length=1, max_length=100)
    hallucination_cases: list[LabeledHallucinationCase] = Field(default_factory=list, max_length=500)
    resolution_cases: list[LabeledResolutionCase] = Field(default_factory=list, max_length=500)


class ClassMetrics(StrictModel):
    total: int
    correct: int
    accuracy: float | None = Field(description="None when the dataset is empty; never 100%")
    confusion: dict[str, dict[str, int]]
    review_count: int = 0


class Mismatch(StrictModel):
    case_id: str
    task: Literal["hallucination", "resolution"]
    expected: str
    actual: str
    category: str | None = None


class EvaluationResponse(StrictModel):
    dataset_name: str
    evaluator_version: str
    hallucination: ClassMetrics
    resolution: ClassMetrics
    mismatches: list[Mismatch]
