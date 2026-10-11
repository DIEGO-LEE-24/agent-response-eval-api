"""Optional prompt-based grounding judge.

Calls an explicitly configured OpenAI-compatible chat endpoint. This path has
mocked unit tests only; no accuracy figures are claimed for it. Two failure
classes are kept apart so that operators can tell "the judge was down" from
"the judge answered badly":

* ``JUDGE_UNAVAILABLE`` - transport/HTTP failure.
* ``JUDGE_INVALID_OUTPUT`` - the model replied, but not with the agreed JSON
  contract (unknown verdict, missing reason, or a citation outside the
  supplied evidence).

Both degrade to ``needs_review``; the judge can never produce an unverified
``grounded``.
"""

import json
import os
from pathlib import Path

import httpx
from fastapi import HTTPException

from app.schemas import HallucinationRequest, HallucinationResponse, Verdict, Violation

PROMPT_VERSION = "grounding-judge-v1"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "grounding_judge_v1.txt"
PROMPT_TEXT = PROMPT_FILE.read_text(encoding="utf-8")
EVALUATOR = "llm-judge"
_ALLOWED_SCHEMES = ("https://", "http://127.0.0.1:", "http://localhost:")


class _InvalidJudgeOutput(ValueError):
    """The remote judge replied, but not within the agreed contract."""


def _fallback(request: HallucinationRequest, code: str, detail: str) -> HallucinationResponse:
    return HallucinationResponse(
        case_id=request.case_id,
        verdict=Verdict.NEEDS_REVIEW,
        violations=[Violation(code=code, detail=detail)],
        evaluator=EVALUATOR,
        prompt_version=PROMPT_VERSION,
        human_review_required=True,
        checked_claims=len(request.claims),
        answer_text_assessed=False,
    )


def _parse_judge_reply(raw: str, allowed_sources: set[str]) -> tuple[Verdict, str]:
    try:
        data = json.loads(raw)
        verdict = Verdict(data["verdict"])
        reason = str(data.get("reason", "")).strip()[:400]
        sources = data.get("supporting_source_ids", [])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _InvalidJudgeOutput("Judge reply is not the agreed JSON object") from exc
    if not reason:
        raise _InvalidJudgeOutput("Judge reply has no reason")
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise _InvalidJudgeOutput("supporting_source_ids must be a list of strings")
    if any(s not in allowed_sources for s in sources):
        raise _InvalidJudgeOutput("Judge cited a source that was not supplied")
    return verdict, reason


async def judge_llm(request: HallucinationRequest) -> HallucinationResponse:
    base = os.getenv("EVAL_LLM_BASE_URL", "").rstrip("/")
    key = os.getenv("EVAL_LLM_API_KEY", "")
    model = os.getenv("EVAL_LLM_MODEL", "")
    if not base or not model:
        raise HTTPException(status_code=503, detail="LLM judge is not configured")
    if not base.startswith(_ALLOWED_SCHEMES):
        raise HTTPException(status_code=400, detail="LLM endpoint must use HTTPS or be localhost")

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": PROMPT_TEXT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "answer": request.answer,
                        "claims": [c.model_dump() for c in request.claims],
                        "evidence": [e.model_dump() for e in request.evidence],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    timeout = float(os.getenv("EVAL_LLM_TIMEOUT_SECONDS", "12"))
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            result = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
            result.raise_for_status()
            body = result.json()
    except (httpx.HTTPError, ValueError):
        return _fallback(
            request, "JUDGE_UNAVAILABLE", "Remote evaluator unreachable or returned an HTTP error"
        )

    try:
        raw = body["choices"][0]["message"]["content"]
        verdict, reason = _parse_judge_reply(raw, {e.source_id for e in request.evidence})
    except (_InvalidJudgeOutput, KeyError, IndexError, TypeError) as exc:
        return _fallback(request, "JUDGE_INVALID_OUTPUT", f"Remote evaluator reply rejected: {exc}")

    violations = [] if verdict == Verdict.GROUNDED else [Violation(code="LLM_JUDGE_FLAG", detail=reason)]
    return HallucinationResponse(
        case_id=request.case_id,
        verdict=verdict,
        violations=violations,
        evaluator=EVALUATOR,
        prompt_version=PROMPT_VERSION,
        human_review_required=verdict != Verdict.GROUNDED,
        checked_claims=len(request.claims),
        answer_text_assessed=True,
    )
