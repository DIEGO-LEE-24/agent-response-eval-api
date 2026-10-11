"""LLM judge tests. The remote model is replaced by httpx.MockTransport.

These tests check request construction and failure handling only. They say
nothing about how accurate a real model would be.
"""

import json

import httpx
import pytest
from fastapi import HTTPException

import app.llm as llm
from app.schemas import HallucinationRequest, Verdict

REQUEST = HallucinationRequest(
    case_id="l",
    answer="Your order shipped.",
    evidence=[{"source_id": "orders:A-1042", "key": "order_status", "value": "shipped"}],
    claims=[{"key": "order_status", "value": "shipped", "cited_source_ids": ["orders:A-1042"]}],
    mode="llm",
)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_BASE_URL", "https://judge.example.test/v1")
    monkeypatch.setenv("EVAL_LLM_MODEL", "test-model")
    monkeypatch.setenv("EVAL_LLM_API_KEY", "test-key")


def use_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(llm.httpx, "AsyncClient", factory)


def reply(content: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


@pytest.mark.asyncio
async def test_not_configured_is_503(monkeypatch):
    monkeypatch.delenv("EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_LLM_MODEL", raising=False)
    with pytest.raises(HTTPException) as exc:
        await llm.judge_llm(REQUEST)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_plain_http_remote_endpoint_is_rejected(monkeypatch, configured):
    monkeypatch.setenv("EVAL_LLM_BASE_URL", "http://judge.example.test/v1")
    with pytest.raises(HTTPException) as exc:
        await llm.judge_llm(REQUEST)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_request_carries_prompt_answer_claims_and_evidence(monkeypatch, configured):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        content = json.dumps({"verdict": "grounded", "reason": "matches", "supporting_source_ids": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    use_transport(monkeypatch, handler)
    await llm.judge_llm(REQUEST)

    assert seen["url"] == "https://judge.example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["temperature"] == 0
    assert body["messages"][0]["content"] == llm.PROMPT_TEXT
    user = json.loads(body["messages"][1]["content"])
    assert user["answer"] == REQUEST.answer
    assert user["claims"][0]["key"] == "order_status"
    assert user["evidence"][0]["source_id"] == "orders:A-1042"


@pytest.mark.asyncio
async def test_valid_reply_is_returned_with_prompt_version(monkeypatch, configured):
    content = json.dumps(
        {
            "verdict": "unsupported",
            "reason": "refund not in evidence",
            "supporting_source_ids": ["orders:A-1042"],
        }
    )
    use_transport(monkeypatch, reply(content))
    result = await llm.judge_llm(REQUEST)
    assert result.verdict == Verdict.UNSUPPORTED
    assert result.prompt_version == llm.PROMPT_VERSION
    assert result.answer_text_assessed is True
    assert [v.code for v in result.violations] == ["LLM_JUDGE_FLAG"]


@pytest.mark.asyncio
async def test_transport_error_is_judge_unavailable(monkeypatch, configured):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    use_transport(monkeypatch, handler)
    result = await llm.judge_llm(REQUEST)
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert [v.code for v in result.violations] == ["JUDGE_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_http_error_status_is_judge_unavailable(monkeypatch, configured):
    use_transport(monkeypatch, lambda request: httpx.Response(500, text="boom"))
    result = await llm.judge_llm(REQUEST)
    assert [v.code for v in result.violations] == ["JUDGE_UNAVAILABLE"]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"verdict": "maybe", "reason": "x"}),
        json.dumps({"verdict": "grounded", "reason": "  "}),
        json.dumps({"verdict": "grounded", "reason": "ok", "supporting_source_ids": "orders:A-1042"}),
        json.dumps({"verdict": "grounded", "reason": "ok", "supporting_source_ids": ["kb:invented"]}),
    ],
    ids=["not-json", "unknown-verdict", "blank-reason", "sources-not-list", "invented-source"],
)
@pytest.mark.asyncio
async def test_contract_violation_is_judge_invalid_output(monkeypatch, configured, content):
    use_transport(monkeypatch, reply(content))
    result = await llm.judge_llm(REQUEST)
    assert result.verdict == Verdict.NEEDS_REVIEW
    assert result.human_review_required is True
    assert [v.code for v in result.violations] == ["JUDGE_INVALID_OUTPUT"]


@pytest.mark.asyncio
async def test_unexpected_response_shape_is_judge_invalid_output(monkeypatch, configured):
    use_transport(monkeypatch, lambda request: httpx.Response(200, json={"choices": []}))
    result = await llm.judge_llm(REQUEST)
    assert [v.code for v in result.violations] == ["JUDGE_INVALID_OUTPUT"]
