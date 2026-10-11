import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import VERSION, app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)

GROUNDING = {
    "case_id": "api-1",
    "answer": "Your order shipped.",
    "evidence": [{"source_id": "orders:A-1042", "key": "order_status", "value": "shipped"}],
    "claims": [{"key": "order_status", "value": "shipped", "cited_source_ids": ["orders:A-1042"]}],
}


def load_dataset() -> dict:
    return json.loads((ROOT / "examples" / "dataset.json").read_text(encoding="utf-8"))


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": VERSION}


def test_grounding_check_response_contract():
    response = client.post("/v1/hallucinations/check", json=GROUNDING)
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "grounded"
    assert body["evaluator"] == "grounding-rules-v2"
    assert body["checked_claims"] == 1
    assert body["answer_text_assessed"] is False
    assert body["prompt_version"] is None


def test_grounding_check_reports_unassessed_answer_text():
    payload = {**GROUNDING, "answer": "Your order shipped. A full refund will be issued tomorrow."}
    body = client.post("/v1/hallucinations/check", json=payload).json()
    assert body["verdict"] == "grounded"
    assert body["answer_text_assessed"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {**GROUNDING, "unexpected": True},
        {**GROUNDING, "mode": "fuzzy"},
        {**GROUNDING, "case_id": ""},
        {**GROUNDING, "claims": [{"key": "order_status", "value": "shipped", "source": "x"}]},
        {k: v for k, v in GROUNDING.items() if k != "answer"},
    ],
    ids=["extra-field", "bad-mode", "empty-case-id", "extra-claim-field", "missing-answer"],
)
def test_grounding_check_rejects_malformed_payloads(payload):
    assert client.post("/v1/hallucinations/check", json=payload).status_code == 422


def test_llm_mode_without_configuration_is_503(monkeypatch):
    monkeypatch.delenv("EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_LLM_MODEL", raising=False)
    response = client.post("/v1/hallucinations/check", json={**GROUNDING, "mode": "llm"})
    assert response.status_code == 503


def test_resolution_check():
    response = client.post(
        "/v1/resolution/check",
        json={
            "case_id": "r-1",
            "customer_confirmed": None,
            "requires_action": True,
            "tool_action_status": "succeeded",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "case_id": "r-1",
        "status": "unknown",
        "reason_code": "NO_CUSTOMER_CONFIRMATION",
        "evaluator": "resolution-rules-v2",
    }


def test_resolution_check_rejects_unknown_action_status():
    response = client.post(
        "/v1/resolution/check",
        json={"case_id": "r-2", "requires_action": True, "tool_action_status": "done"},
    )
    assert response.status_code == 422


def test_evaluation_run_matches_recorded_baseline():
    baseline = json.loads((ROOT / "examples" / "baseline.json").read_text(encoding="utf-8"))
    response = client.post("/v1/evaluations/run", json=load_dataset())
    assert response.status_code == 200
    body = response.json()
    assert body["hallucination"]["correct"] == baseline["hallucination_correct"]
    assert body["hallucination"]["total"] == baseline["hallucination_total"]
    assert body["resolution"]["correct"] == baseline["resolution_correct"]
    assert body["resolution"]["total"] == baseline["resolution_total"]
    assert sorted(m["case_id"] for m in body["mismatches"]) == baseline["known_mismatches"]
    assert all(m["category"] for m in body["mismatches"])


def test_evaluation_run_rejects_llm_cases():
    dataset = load_dataset()
    dataset["hallucination_cases"][0]["input"]["mode"] = "llm"
    assert client.post("/v1/evaluations/run", json=dataset).status_code == 422


def test_empty_evaluation_reports_no_accuracy():
    body = client.post("/v1/evaluations/run", json={"dataset_name": "empty"}).json()
    assert body["hallucination"]["accuracy"] is None
    assert body["resolution"]["accuracy"] is None
    assert body["mismatches"] == []


def test_dataset_has_unique_ids_and_no_duplicate_resolution_scenarios():
    dataset = load_dataset()
    h_ids = [c["input"]["case_id"] for c in dataset["hallucination_cases"]]
    r_ids = [c["input"]["case_id"] for c in dataset["resolution_cases"]]
    assert len(h_ids) == len(set(h_ids))
    assert len(r_ids) == len(set(r_ids))
    scenarios = [
        json.dumps({k: v for k, v in c["input"].items() if k != "case_id"}, sort_keys=True)
        for c in dataset["resolution_cases"]
    ]
    assert len(scenarios) == len(set(scenarios))
    assert all(
        c["category"] and c["note"] for c in dataset["hallucination_cases"] + dataset["resolution_cases"]
    )
