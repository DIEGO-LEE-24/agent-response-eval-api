"""Render docs/assets/grounding-eval.svg from the current evaluation results.

    python scripts/make_eval_chart.py

Every number in the chart comes from evaluate() over examples/dataset.json;
nothing is typed in by hand. Re-run after changing the rules or the dataset.
"""

import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate  # noqa: E402
from app.schemas import EvaluationRequest, Verdict  # noqa: E402

DATASET = ROOT / "examples" / "dataset.json"
OUT = ROOT / "docs" / "assets" / "grounding-eval.svg"

W = 1200
LABELS = [v.value for v in Verdict]
GREEN, RED, AMBER = ("#1a7f37", "#dafbe1"), ("#cf222e", "#ffebe9"), ("#9a6700", "#fff8c5")
EMPTY = ("#d0d7de", "#f6f8fa")
SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
STYLE = (
    f"text {{ font-family: {SANS}; fill: #1f2328; }}"
    " .mono { font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace; }"
    " .h { font-size: 15.5px; font-weight: 600; }"
    " .b { font-size: 13.5px; fill: #424a53; }"
    " .s { font-size: 12.5px; fill: #57606a; }"
)


def text(x: float, y: float, body: str, cls: str = "b", anchor: str = "start", style: str = "") -> str:
    extra = f' style="{style}"' if style else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{extra}>{escape(body)}</text>'


def rect(x: float, y: float, w: float, h: float, rx: float, colors: tuple[str, str]) -> str:
    stroke, fill = colors
    return f'<rect x="{x}" y="{y}" width="{w:.0f}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def chips(
    x0: float, y0: float, width: float, items: list[str], colors: tuple[str, str]
) -> tuple[list[str], float]:
    """Lay out labelled pills left to right, wrapping inside ``width``; return the bottom edge."""
    out, x, y = [], x0, y0
    for item in items:
        w = 6.3 * len(item) + 24
        if x + w > x0 + width:
            x, y = x0, y + 32
        out.append(rect(x, y, w, 24, 12, colors))
        out.append(text(x + w / 2, y + 16.5, item, "s", "middle", f"fill:{colors[0]}"))
        x += w + 8
    return out, y + 24


def main() -> int:
    dataset = EvaluationRequest.model_validate_json(DATASET.read_text(encoding="utf-8"))
    result = evaluate(dataset)
    g, r = result.hallucination, result.resolution
    grounding = [m for m in result.mismatches if m.task == "hallucination"]
    rejected = [m for m in grounding if m.actual == "unsupported"]
    passed = [m for m in grounding if m.actual == "grounded"]
    other = [m for m in grounding if m.actual not in ("unsupported", "grounded")]

    s: list[str] = [
        text(
            32,
            44,
            f"Grounding evaluation · {dataset.dataset_name}",
            "h",
            style="font-size:22px;font-weight:700",
        ),
        text(
            32,
            72,
            f"{g.correct} / {g.total} agree with label ({g.accuracy:.1%})  ·  "
            f"{len(rejected)} supported answers rejected  ·  {len(passed)} wrongly passed",
            style="font-size:16px;fill:#1f2328",
        ),
    ]

    # Confusion matrix: rows = expected label, columns = actual verdict.
    cx, cy, cw, ch = 196, 150, 112, 66
    s.append(text(cx + cw * 1.5, 112, "actual verdict →", "s", "middle"))
    s.append(text(cx - 14, cy - 12, "expected ↓", "s", "end"))
    for j, actual in enumerate(LABELS):
        s.append(text(cx + cw * j + cw / 2, cy - 12, actual, "mono s", "middle"))
    for i, expected in enumerate(LABELS):
        y = cy + ch * i
        row_total = sum(g.confusion[expected].values())
        s.append(text(cx - 14, y + ch / 2 + 5, f"{expected} ({row_total})", "mono s", "end"))
        for j, actual in enumerate(LABELS):
            n = g.confusion[expected][actual]
            if not n:
                colors = EMPTY
            elif expected == actual:
                colors = GREEN
            elif actual == "grounded":
                colors = AMBER
            else:
                colors = RED
            x = cx + cw * j
            s.append(rect(x + 3, y + 3, cw - 6, ch - 6, 6, colors))
            color = colors[0] if n else "#8c959f"
            s.append(text(x + cw / 2, y + ch / 2 + 8, str(n), "h", "middle", f"font-size:22px;fill:{color}"))
    legend_y = cy + ch * 3 + 30
    for lx, label, colors in [
        (40, "agrees", GREEN),
        (150, "rejected a supported answer", RED),
        (380, "passed as grounded", AMBER),
    ]:
        s.append(rect(lx, legend_y - 12, 14, 14, 3, colors))
        s.append(text(lx + 20, legend_y, label, "s"))

    # Mismatch breakdown, grouped by direction of the error.
    px, pw = 600, 552
    panel: list[str] = [
        text(px, 128, f"Supported answer flagged unsupported · {len(rejected)}", "h", style=f"fill:{RED[0]}"),
        text(px, 148, "values and keys are compared as strings after light normalization", "s"),
    ]
    out, y = chips(px, 160, pw, [f"{m.case_id} {m.category}" for m in rejected], RED)
    panel += out
    y += 36
    title = f"Unsupported or conflicting answer passed as grounded · {len(passed)}"
    panel.append(text(px, y, title, "h", style=f"fill:{AMBER[0]}"))
    panel.append(text(px, y + 20, "answer text is not assessed; caller-extracted claims are trusted", "s"))
    out, y = chips(px, y + 32, pw, [f"{m.case_id} {m.category}" for m in passed], AMBER)
    panel += out
    if other:
        panel.append(text(px, y + 36, f"Other mismatches · {len(other)}", "h"))
        out, y = chips(px, y + 48, pw, [f"{m.case_id} {m.category}" for m in other], RED)
        panel += out
    panel.append(text(px, y + 30, "All mismatch IDs are pinned in examples/baseline.json.", "s"))
    panel_bottom = y + 48
    s.append(rect(px - 16, 100, pw + 32, panel_bottom - 100, 8, EMPTY))
    s += panel

    notes_y = max(panel_bottom, legend_y) + 44
    notes = [
        f"{g.total} hand-written synthetic cases. These numbers describe the rules on this set only; "
        "they are not hallucination-detection accuracy on real agent traffic.",
        f"Resolution: {r.correct} / {r.total} synthetic policy scenarios agree. Labels follow the documented "
        "policy, so this is not a real-conversation resolution rate.",
    ]
    for k, note in enumerate(notes):
        s.append(text(32, notes_y + 22 * k, note))
    s.append(
        text(32, notes_y + 48, "Generated by scripts/make_eval_chart.py from examples/dataset.json.", "s")
    )
    h = notes_y + 72

    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        'role="img" aria-labelledby="t">',
        f'<title id="t">Grounding evaluation: {g.correct} of {g.total} synthetic cases agree with the label'
        "</title>",
        f"<defs><style>{STYLE}</style></defs>",
        rect(0.5, 0.5, W - 1, h - 1, 12, EMPTY[:1] + ("#ffffff",)),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join([*head, *s, "</svg>"]) + "\n", encoding="utf-8")
    print(f"written: {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
