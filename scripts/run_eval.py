"""Offline regression evaluation.

    python scripts/run_eval.py                  # print the evaluation report
    python scripts/run_eval.py --check-baseline # exit 1 if results drift from examples/baseline.json
    python scripts/run_eval.py --write-baseline # record the current results as the new baseline

The baseline stores the known mismatches by case ID, so a regression is detected
both when a previously correct case breaks and when a documented limitation is
silently "fixed" without updating the record.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate  # noqa: E402
from app.schemas import EvaluationRequest  # noqa: E402

DATASET = ROOT / "examples" / "dataset.json"
BASELINE = ROOT / "examples" / "baseline.json"


def main(argv: list[str]) -> int:
    dataset = EvaluationRequest.model_validate_json(DATASET.read_text(encoding="utf-8"))
    result = evaluate(dataset)
    summary = {
        "evaluator_version": result.evaluator_version,
        "hallucination_correct": result.hallucination.correct,
        "hallucination_total": result.hallucination.total,
        "resolution_correct": result.resolution.correct,
        "resolution_total": result.resolution.total,
        "known_mismatches": sorted(m.case_id for m in result.mismatches),
    }

    if "--write-baseline" in argv:
        BASELINE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE}")
        return 0

    print(result.model_dump_json(indent=2))
    if "--check-baseline" in argv:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        if summary != baseline:
            print("Regression: results differ from examples/baseline.json", file=sys.stderr)
            print(json.dumps({"baseline": baseline, "current": summary}, indent=2), file=sys.stderr)
            return 1
        print("baseline check: ok", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
