"""Deterministic evaluators.

Design choices, in order of importance:

1. Fail closed. Anything the rules cannot verify becomes ``needs_review`` or
   ``unsupported``; ``grounded`` is only returned when every supplied claim
   matches supplied evidence.
2. Be explicit about scope. The grounding evaluator compares *structured claims*
   with *structured evidence*. It does not read the natural-language answer, so
   it cannot catch statements that were never extracted as claims. That gap is
   covered by a regression test instead of being hidden.
3. Prefer a conservative string comparison over fuzzy matching. Synonyms,
   currency formats and unit suffixes are therefore reported as mismatches; the
   dataset records these as known false positives rather than tuning them away.
"""

import re

from app.schemas import (
    AnswerClaim,
    EvidenceFact,
    HallucinationRequest,
    HallucinationResponse,
    ResolutionRequest,
    ResolutionResponse,
    ResolutionStatus,
    Verdict,
    Violation,
)

GROUNDING_EVALUATOR_VERSION = "grounding-rules-v2"
RESOLUTION_EVALUATOR_VERSION = "resolution-rules-v2"

# Codes that make a claim unsupported on their own.
HARD_CODES = frozenset({"MISSING_EVIDENCE", "VALUE_MISMATCH", "UNKNOWN_SOURCE", "CITATION_MISMATCH"})
# Codes that only mean "a human has to look at this".
SOFT_CODES = frozenset({"EMPTY_ANSWER", "NO_EXTRACTED_CLAIMS", "CONFLICTING_EVIDENCE"})

_EDGE_PUNCTUATION = re.compile(r"^[\s\"'“”‘’.,;:!?()\[\]]+|[\s\"'“”‘’.,;:!?()\[\]]+$")
_INNER_SPACE = re.compile(r"\s+")


def normalize(value: str) -> str:
    """Whitespace, casing and edge punctuation only.

    Deliberately narrow: ``"$10"`` and ``"10 USD"`` stay different, and so do
    ``"shipped"`` and ``"dispatched"``. Semantic equivalence is out of scope for
    the rules evaluator.
    """
    stripped = _EDGE_PUNCTUATION.sub("", value)
    return _INNER_SPACE.sub(" ", stripped).strip().casefold()


def _facts_for_key(claim: AnswerClaim, facts: list[EvidenceFact]) -> list[EvidenceFact]:
    key = normalize(claim.key)
    return [f for f in facts if normalize(f.key) == key]


def _check_claim(claim: AnswerClaim, facts: list[EvidenceFact], source_ids: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    matches = _facts_for_key(claim, facts)
    claim_value = normalize(claim.value)

    if not matches:
        violations.append(
            Violation(
                code="MISSING_EVIDENCE",
                claim_key=claim.key,
                detail="No evidence fact exists for this claim key",
            )
        )
    else:
        distinct_values = {normalize(f.value) for f in matches}
        cited_match = any(
            f.source_id in claim.cited_source_ids and normalize(f.value) == claim_value for f in matches
        )
        if claim_value not in distinct_values:
            violations.append(
                Violation(
                    code="VALUE_MISMATCH",
                    claim_key=claim.key,
                    detail="Claim value is not identical to any evidence value after normalization",
                )
            )
        elif len(distinct_values) > 1 and not cited_match:
            # Evidence disagrees with itself and the claim does not say which
            # source it relied on. Do not guess; ask a human.
            violations.append(
                Violation(
                    code="CONFLICTING_EVIDENCE",
                    claim_key=claim.key,
                    detail="Evidence sources disagree for this key and the claim cites no matching source",
                )
            )

    for sid in claim.cited_source_ids:
        if sid not in source_ids:
            violations.append(
                Violation(
                    code="UNKNOWN_SOURCE",
                    claim_key=claim.key,
                    detail=f"Cited source ID is absent from supplied evidence: {sid}",
                )
            )
        elif not any(f.source_id == sid and normalize(f.value) == claim_value for f in matches):
            violations.append(
                Violation(
                    code="CITATION_MISMATCH",
                    claim_key=claim.key,
                    detail=f"Cited source does not contain a matching key/value: {sid}",
                )
            )
    return violations


def judge_hallucination(request: HallucinationRequest) -> HallucinationResponse:
    """Verify structured claims against structured evidence.

    Returns ``grounded`` only when every claim matches a supplied fact and every
    citation points at a fact with that value. The free-text ``answer`` is not
    analysed beyond an emptiness check.
    """
    violations: list[Violation] = []
    if not request.answer.strip():
        violations.append(Violation(code="EMPTY_ANSWER", detail="Answer contains no text"))
    if not request.claims:
        violations.append(
            Violation(code="NO_EXTRACTED_CLAIMS", detail="No claims supplied; nothing could be verified")
        )

    source_ids = {f.source_id for f in request.evidence}
    for claim in request.claims:
        violations.extend(_check_claim(claim, request.evidence, source_ids))

    codes = {v.code for v in violations}
    if codes & HARD_CODES:
        verdict = Verdict.UNSUPPORTED
    elif codes & SOFT_CODES:
        verdict = Verdict.NEEDS_REVIEW
    else:
        verdict = Verdict.GROUNDED

    return HallucinationResponse(
        case_id=request.case_id,
        verdict=verdict,
        violations=violations,
        evaluator=GROUNDING_EVALUATOR_VERSION,
        human_review_required=verdict != Verdict.GROUNDED,
        checked_claims=len(request.claims),
        answer_text_assessed=False,
    )


def judge_resolution(request: ResolutionRequest) -> ResolutionResponse:
    """Classify whether a conversation is resolved.

    Priority order:
    1. A hand-off is terminal.
    2. No stated goal means nothing can be called resolved.
    3. An explicit customer rejection is terminal, whatever the tools did.
    4. Required actions must have succeeded; failed is unresolved, anything
       else is not yet verifiable.
    5. Contradictory inputs (a failed/pending action on a case that needed no
       action, or "not_required" on a case that did) are never resolved.
    6. Only then does explicit confirmation count as resolved. Silence is
       ``unknown``, not success.
    """
    action = request.tool_action_status
    if request.handed_off:
        status, reason = ResolutionStatus.HANDOFF, "HUMAN_HANDOFF"
    elif not request.customer_goal_present:
        status, reason = ResolutionStatus.UNKNOWN, "GOAL_UNSPECIFIED"
    elif request.customer_confirmed is False:
        status, reason = ResolutionStatus.UNRESOLVED, "CUSTOMER_REJECTED"
    elif request.requires_action and action == "failed":
        status, reason = ResolutionStatus.UNRESOLVED, "ACTION_FAILED"
    elif request.requires_action and action == "not_required":
        status, reason = ResolutionStatus.UNKNOWN, "INCONSISTENT_ACTION_SIGNAL"
    elif request.requires_action and action != "succeeded":
        status, reason = ResolutionStatus.UNKNOWN, "ACTION_NOT_VERIFIED"
    elif not request.requires_action and action in ("failed", "pending"):
        status, reason = ResolutionStatus.UNKNOWN, "INCONSISTENT_ACTION_SIGNAL"
    elif request.customer_confirmed is True:
        status, reason = ResolutionStatus.RESOLVED, "CONFIRMED_AND_VERIFIED"
    else:
        status, reason = ResolutionStatus.UNKNOWN, "NO_CUSTOMER_CONFIRMATION"
    return ResolutionResponse(
        case_id=request.case_id, status=status, reason_code=reason, evaluator=RESOLUTION_EVALUATOR_VERSION
    )
