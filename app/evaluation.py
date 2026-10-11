"""Regression evaluation over a labeled dataset.

The confusion matrices use the *expected* label as rows and the *actual*
verdict as columns, so a false positive of the rules evaluator shows up as
``expected=grounded, actual=unsupported``.
"""

from app.rules import (
    GROUNDING_EVALUATOR_VERSION,
    RESOLUTION_EVALUATOR_VERSION,
    judge_hallucination,
    judge_resolution,
)
from app.schemas import (
    ClassMetrics,
    EvaluationRequest,
    EvaluationResponse,
    Mismatch,
    ResolutionStatus,
    Verdict,
)


def _metrics(
    pairs: list[tuple[str, str]], labels: list[str], review_label: str | None = None
) -> ClassMetrics:
    confusion = {expected: dict.fromkeys(labels, 0) for expected in labels}
    for expected, actual in pairs:
        confusion[expected][actual] += 1
    total = len(pairs)
    correct = sum(expected == actual for expected, actual in pairs)
    return ClassMetrics(
        total=total,
        correct=correct,
        accuracy=(correct / total) if total else None,
        confusion=confusion,
        review_count=sum(actual == review_label for _, actual in pairs) if review_label else 0,
    )


def evaluate(dataset: EvaluationRequest) -> EvaluationResponse:
    h_pairs: list[tuple[str, str]] = []
    r_pairs: list[tuple[str, str]] = []
    mismatches: list[Mismatch] = []

    for case in dataset.hallucination_cases:
        actual = judge_hallucination(case.input).verdict.value
        expected = case.expected_verdict.value
        h_pairs.append((expected, actual))
        if expected != actual:
            mismatches.append(
                Mismatch(
                    case_id=case.input.case_id,
                    task="hallucination",
                    expected=expected,
                    actual=actual,
                    category=case.category,
                )
            )

    for case in dataset.resolution_cases:
        actual = judge_resolution(case.input).status.value
        expected = case.expected_status.value
        r_pairs.append((expected, actual))
        if expected != actual:
            mismatches.append(
                Mismatch(
                    case_id=case.input.case_id,
                    task="resolution",
                    expected=expected,
                    actual=actual,
                    category=case.category,
                )
            )

    return EvaluationResponse(
        dataset_name=dataset.dataset_name,
        evaluator_version=f"{GROUNDING_EVALUATOR_VERSION}+{RESOLUTION_EVALUATOR_VERSION}",
        hallucination=_metrics(h_pairs, [v.value for v in Verdict], Verdict.NEEDS_REVIEW.value),
        resolution=_metrics(r_pairs, [s.value for s in ResolutionStatus]),
        mismatches=mismatches,
    )
