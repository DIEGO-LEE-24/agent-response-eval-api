import pytest

from app.rules import judge_hallucination, judge_resolution, normalize
from app.schemas import HallucinationRequest, ResolutionRequest, Verdict

ORDER = {"source_id": "orders:A-1042", "key": "order_status", "value": "shipped"}


def check(answer="Your order shipped.", evidence=None, claims=None):
    return judge_hallucination(
        HallucinationRequest(
            case_id="t",
            answer=answer,
            evidence=[ORDER] if evidence is None else evidence,
            claims=[{"key": "order_status", "value": "shipped", "cited_source_ids": ["orders:A-1042"]}]
            if claims is None
            else claims,
        )
    )


def codes(response):
    return [v.code for v in response.violations]


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  SHIPPED. ", "shipped"), ('"true"', "true"), ("next  business   day", "next business day")],
)
def test_normalize_only_touches_case_whitespace_and_edge_punctuation(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(("a", "b"), [("$10.00", "10 USD"), ("dispatched", "shipped"), ("1,200", "1200")])
def test_normalize_does_not_claim_semantic_equivalence(a, b):
    assert normalize(a) != normalize(b)


# --------------------------------------------------------------------------- #
# grounding
# --------------------------------------------------------------------------- #


def test_matching_claim_with_matching_citation_is_grounded():
    result = check()
    assert result.verdict == Verdict.GROUNDED
    assert result.violations == []
    assert result.human_review_required is False
    assert result.checked_claims == 1


def test_citation_is_optional_when_evidence_is_consistent():
    result = check(claims=[{"key": "order_status", "value": "shipped"}])
    assert result.verdict == Verdict.GROUNDED


def test_wrong_value_is_unsupported():
    result = check(claims=[{"key": "order_status", "value": "delivered"}])
    assert result.verdict == Verdict.UNSUPPORTED
    assert codes(result) == ["VALUE_MISMATCH"]


def test_claim_without_any_evidence_is_unsupported():
    result = check(claims=[{"key": "shipping_fee", "value": "0"}])
    assert result.verdict == Verdict.UNSUPPORTED
    assert codes(result) == ["MISSING_EVIDENCE"]


def test_citation_to_unsupplied_source_is_unsupported():
    result = check(claims=[{"key": "order_status", "value": "shipped", "cited_source_ids": ["kb:999"]}])
    assert result.verdict == Verdict.UNSUPPORTED
    assert codes(result) == ["UNKNOWN_SOURCE"]


def test_citation_to_source_without_that_fact_is_unsupported():
    evidence = [ORDER, {"source_id": "billing:INV-1", "key": "invoice_status", "value": "paid"}]
    claims = [{"key": "order_status", "value": "shipped", "cited_source_ids": ["billing:INV-1"]}]
    result = check(evidence=evidence, claims=claims)
    assert result.verdict == Verdict.UNSUPPORTED
    assert codes(result) == ["CITATION_MISMATCH"]


def test_one_bad_claim_makes_the_whole_answer_unsupported():
    claims = [{"key": "order_status", "value": "shipped"}, {"key": "order_status", "value": "lost"}]
    assert check(claims=claims).verdict == Verdict.UNSUPPORTED


CONFLICT = [
    {"source_id": "orders:D-1", "key": "delivery_date", "value": "2026-03-05"},
    {"source_id": "carrier:T-1", "key": "delivery_date", "value": "2026-03-06"},
]


def test_conflicting_evidence_without_citation_needs_review():
    result = check(evidence=CONFLICT, claims=[{"key": "delivery_date", "value": "2026-03-05"}])
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert codes(result) == ["CONFLICTING_EVIDENCE"]


def test_conflicting_evidence_claim_matching_neither_is_unsupported():
    result = check(evidence=CONFLICT, claims=[{"key": "delivery_date", "value": "2026-03-08"}])
    assert result.verdict == Verdict.UNSUPPORTED


def test_known_gap_citing_one_side_of_a_conflict_passes():
    """Dataset case G-25: a reviewer wants needs_review, the rules return grounded."""
    claims = [{"key": "delivery_date", "value": "2026-03-05", "cited_source_ids": ["orders:D-1"]}]
    assert check(evidence=CONFLICT, claims=claims).verdict == Verdict.GROUNDED


def test_empty_answer_needs_review_even_if_claims_match():
    result = check(answer="   ")
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert codes(result) == ["EMPTY_ANSWER"]


def test_no_claims_needs_review():
    result = check(answer="Everything looks fine.", claims=[])
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert codes(result) == ["NO_EXTRACTED_CLAIMS"]
    assert result.checked_claims == 0


def test_hard_violation_outranks_soft_violation():
    result = check(answer="", claims=[{"key": "order_status", "value": "lost"}])
    assert set(codes(result)) == {"EMPTY_ANSWER", "VALUE_MISMATCH"}
    assert result.verdict == Verdict.UNSUPPORTED


def test_known_gap_unextracted_statement_in_answer_text_is_not_judged():
    """Regression guard for the documented scope limit (dataset case G-27).

    The answer promises a refund that no evidence supports. Because only the
    order-status claim was extracted, the rules evaluator returns grounded and
    says so via answer_text_assessed=False. If this test starts failing, the
    README, API docs and baseline must be updated together.
    """
    result = check(answer="Your order shipped. A full refund will be issued tomorrow.")
    assert result.verdict == Verdict.GROUNDED
    assert result.answer_text_assessed is False
    assert result.checked_claims == 1


def test_known_gap_synonym_is_reported_as_mismatch():
    """Dataset case G-08: conservative comparison produces a false positive."""
    result = check(claims=[{"key": "order_status", "value": "dispatched"}])
    assert result.verdict == Verdict.UNSUPPORTED


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("inputs", "status", "reason"),
    [
        ({"customer_confirmed": True, "requires_action": False}, "resolved", "CONFIRMED_AND_VERIFIED"),
        (
            {"customer_confirmed": True, "requires_action": True, "tool_action_status": "succeeded"},
            "resolved",
            "CONFIRMED_AND_VERIFIED",
        ),
        ({"customer_confirmed": None, "requires_action": False}, "unknown", "NO_CUSTOMER_CONFIRMATION"),
        (
            {"customer_confirmed": False, "requires_action": True, "tool_action_status": "succeeded"},
            "unresolved",
            "CUSTOMER_REJECTED",
        ),
        (
            {"customer_confirmed": True, "requires_action": True, "tool_action_status": "failed"},
            "unresolved",
            "ACTION_FAILED",
        ),
        (
            {"customer_confirmed": True, "requires_action": True, "tool_action_status": "pending"},
            "unknown",
            "ACTION_NOT_VERIFIED",
        ),
        (
            {"customer_confirmed": True, "requires_action": True, "tool_action_status": "not_attempted"},
            "unknown",
            "ACTION_NOT_VERIFIED",
        ),
        (
            {"customer_confirmed": True, "requires_action": True, "tool_action_status": "not_required"},
            "unknown",
            "INCONSISTENT_ACTION_SIGNAL",
        ),
        (
            {"customer_confirmed": True, "requires_action": False, "tool_action_status": "failed"},
            "unknown",
            "INCONSISTENT_ACTION_SIGNAL",
        ),
        (
            {"customer_confirmed": True, "requires_action": False, "customer_goal_present": False},
            "unknown",
            "GOAL_UNSPECIFIED",
        ),
        (
            {"customer_confirmed": True, "requires_action": False, "handed_off": True},
            "handoff",
            "HUMAN_HANDOFF",
        ),
    ],
)
def test_resolution_policy(inputs, status, reason):
    result = judge_resolution(ResolutionRequest(case_id="r", **inputs))
    assert (result.status.value, result.reason_code) == (status, reason)


def test_rejection_takes_precedence_over_action_state():
    result = judge_resolution(
        ResolutionRequest(
            case_id="r", customer_confirmed=False, requires_action=True, tool_action_status="pending"
        )
    )
    assert result.reason_code == "CUSTOMER_REJECTED"


def test_handoff_takes_precedence_over_everything():
    result = judge_resolution(
        ResolutionRequest(
            case_id="r",
            customer_confirmed=False,
            requires_action=True,
            tool_action_status="failed",
            customer_goal_present=False,
            handed_off=True,
        )
    )
    assert result.status.value == "handoff"
