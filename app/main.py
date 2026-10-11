"""HTTP surface: health, structured grounding check, resolution check, dataset evaluation."""

from fastapi import FastAPI, HTTPException

from app.evaluation import evaluate
from app.llm import judge_llm
from app.rules import judge_hallucination, judge_resolution
from app.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    HallucinationRequest,
    HallucinationResponse,
    ResolutionRequest,
    ResolutionResponse,
)

VERSION = "0.2.0"

app = FastAPI(
    title="Agent Response Evaluation API",
    description=(
        "Checks agent answers against structured evidence, classifies conversation "
        "resolution conservatively, and runs regression evaluations over a labeled dataset. "
        "The rules evaluator verifies caller-extracted claims; it does not analyse free text."
    ),
    version=VERSION,
)


@app.get("/health", summary="Liveness")
def health() -> dict[str, str]:
    return {"status": "ok", "version": VERSION}


@app.post(
    "/v1/hallucinations/check",
    response_model=HallucinationResponse,
    summary="Structured claim grounding check",
    description=(
        "mode=rules (default): verifies each supplied claim against supplied evidence and "
        "citations. Statements in the answer that were not extracted as claims are not judged "
        "(answer_text_assessed=false). mode=llm: forwards answer, claims and evidence to a "
        "configured OpenAI-compatible judge; any transport or contract failure yields needs_review."
    ),
)
async def hallucination_check(request: HallucinationRequest) -> HallucinationResponse:
    if request.mode == "llm":
        return await judge_llm(request)
    return judge_hallucination(request)


@app.post(
    "/v1/resolution/check",
    response_model=ResolutionResponse,
    summary="Conservative resolution classification",
    description=(
        "Resolution requires an explicit customer confirmation and, when the request needed "
        "an action, a verified successful action. Absence of a hand-off is not treated as success."
    ),
)
def resolution_check(request: ResolutionRequest) -> ResolutionResponse:
    return judge_resolution(request)


@app.post(
    "/v1/evaluations/run",
    response_model=EvaluationResponse,
    summary="Regression evaluation over a labeled dataset",
    description="Runs the deterministic evaluators only; mode=llm cases are rejected with 422.",
)
def evaluations_run(request: EvaluationRequest) -> EvaluationResponse:
    if any(case.input.mode != "rules" for case in request.hallucination_cases):
        raise HTTPException(status_code=422, detail="Dataset evaluations require mode=rules")
    return evaluate(request)
